"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Create sessions table
    op.create_table(
        'sessions',
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('module', sa.String(), nullable=False),
        sa.Column('start_time', sa.DateTime(), nullable=False),
        sa.Column('end_time', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('session_id')
    )
    op.create_index('ix_sessions_user_id', 'sessions', ['user_id'])

    # Create responses table
    op.create_table(
        'responses',
        sa.Column('response_id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('question_id', sa.String(), nullable=False),
        sa.Column('response_text', sa.String(), nullable=False),
        sa.Column('response_value', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.session_id'], ),
        sa.PrimaryKeyConstraint('response_id')
    )

    # Create analyses table
    op.create_table(
        'analyses',
        sa.Column('analysis_id', sa.String(), nullable=False),
        sa.Column('analyzer_id', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('results', postgresql.JSON(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('recommendations', postgresql.JSON(), nullable=False),
        sa.PrimaryKeyConstraint('analysis_id')
    )

    # Create session_analysis association table
    op.create_table(
        'session_analysis',
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('analysis_id', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['analysis_id'], ['analyses.analysis_id'], ),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.session_id'], )
    )

    # Create profiles table
    op.create_table(
        'profiles',
        sa.Column('profile_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('demographics', postgresql.JSON(), nullable=False),
        sa.Column('attachment_style', sa.String(), nullable=False),
        sa.Column('communication_style', sa.String(), nullable=False),
        sa.Column('conflict_style', sa.String(), nullable=False),
        sa.Column('family_patterns', postgresql.JSON(), nullable=False),
        sa.Column('relationship_goals', postgresql.JSON(), nullable=False),
        sa.Column('health_indicators', postgresql.JSON(), nullable=False),
        sa.Column('risk_factors', postgresql.JSON(), nullable=False),
        sa.Column('strengths', postgresql.JSON(), nullable=False),
        sa.Column('areas_for_growth', postgresql.JSON(), nullable=False),
        sa.PrimaryKeyConstraint('profile_id')
    )
    op.create_index('ix_profiles_user_id', 'profiles', ['user_id'])

    # Create assessments table
    op.create_table(
        'assessments',
        sa.Column('assessment_id', sa.String(), nullable=False),
        sa.Column('profile_id', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('scoring_metrics', postgresql.JSON(), nullable=False),
        sa.Column('recommendations', postgresql.JSON(), nullable=False),
        sa.Column('next_steps', postgresql.JSON(), nullable=False),
        sa.Column('review_date', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['profile_id'], ['profiles.profile_id'], ),
        sa.PrimaryKeyConstraint('assessment_id')
    )

    # Create questions table
    op.create_table(
        'questions',
        sa.Column('question_id', sa.String(), nullable=False),
        sa.Column('question_text', sa.String(), nullable=False),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('module', sa.String(), nullable=False),
        sa.Column('framework', sa.String(), nullable=True),
        sa.Column('domain', sa.String(), nullable=True),
        sa.Column('metadata', postgresql.JSON(), nullable=False),
        sa.PrimaryKeyConstraint('question_id')
    )

def downgrade() -> None:
    op.drop_table('questions')
    op.drop_table('assessments')
    op.drop_table('profiles')
    op.drop_table('session_analysis')
    op.drop_table('analyses')
    op.drop_table('responses')
    op.drop_table('sessions')