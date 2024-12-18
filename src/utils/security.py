from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from .config import config

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    secret_key = config.get_value("security", "jwt_secret_key")
    algorithm = config.get_value("security", "jwt_algorithm", default="HS256")
    
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        secret_key = config.get_value("security", "jwt_secret_key")
        algorithm = config.get_value("security", "jwt_algorithm", default="HS256")
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=payload.get("role"))
    except JWTError:
        raise credentials_exception
    
    # Here you would typically get the user from your database
    # This is a placeholder implementation
    user = User(
        username=token_data.username,
        role=token_data.role or "user",
        disabled=False
    )
    
    if user is None:
        raise credentials_exception
    return user

def check_permissions(required_role: str):
    async def permission_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user
    return permission_checker

class SecurityManager:
    def __init__(self):
        self.pwd_context = pwd_context
        self.oauth2_scheme = oauth2_scheme

    def create_user(self, username: str, password: str, role: str = "user") -> UserInDB:
        hashed_password = self.pwd_context.hash(password)
        return UserInDB(
            username=username,
            role=role,
            hashed_password=hashed_password
        )

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        # Here you would typically get the user from your database
        # This is a placeholder implementation
        user = UserInDB(
            username=username,
            role="user",
            hashed_password=get_password_hash(password)
        )
        
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return User(
            username=user.username,
            role=user.role,
            disabled=False
        )

    def create_user_token(self, user: User) -> Token:
        access_token_expires = timedelta(
            minutes=config.get_value("security", "access_token_expire_minutes", default=30)
        )
        access_token = create_access_token(
            data={"sub": user.username, "role": user.role},
            expires_delta=access_token_expires
        )
        return Token(access_token=access_token, token_type="bearer")

    def encrypt_data(self, data: str) -> str:
        # Implement encryption logic here
        # This is a placeholder
        return data

    def decrypt_data(self, encrypted_data: str) -> str:
        # Implement decryption logic here
        # This is a placeholder
        return encrypted_data

security_manager = SecurityManager()