-- Enable UUID extension
create extension if not exists "uuid-ossp";

-- Users table
create table users (
    id uuid default uuid_generate_v4() primary key,
    username text not null,
    email text unique not null,
    password_hash text not null,
    created_at timestamptz default now()
);

-- Predictions table with all cancer markers
create table predictions (
    id uuid default uuid_generate_v4() primary key,
    user_id uuid references users(id),
    age float not null,
    CA125 float not null,
    HE4 float not null,
    CA19_9 float not null,
    AFP float not null,
    GGT float not null,
    CEA float not null,
    HGB float not null,
    ALP float not null,
    CA72_4 float not null,
    Ca float not null,
    menopausal_status text not null,
    family_history boolean not null,
    smoking_status text not null,
    alcohol boolean not null,
    cancer boolean,
    prediction_result float not null,
    created_at timestamptz default now()
);

-- Create indexes for better query performance
create index idx_predictions_user_id on predictions(user_id);
create index idx_predictions_created_at on predictions(created_at);
