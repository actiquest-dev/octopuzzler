#!/usr/bin/env python3
"""User management CLI tool"""

import asyncio
import click
from face_recognition_service import FaceRecognitionService

face_rec = FaceRecognitionService()

@click.group()
def cli():
    """User management commands"""
    pass

@cli.command()
async def list_users():
    """List all registered users"""
    users = await face_rec.list_users()
    
    click.echo(f"\nRegistered Users ({len(users)}):\n")
    click.echo(f"{'ID':<12} {'Name':<20} {'Sessions':<10} {'Last Seen'}")
    click.echo("-" * 70)
    
    for user in users:
        click.echo(
            f"{user['user_id']:<12} "
            f"{user['name']:<20} "
            f"{user['total_sessions']:<10} "
            f"{user['last_seen']}"
        )

@cli.command()
@click.argument('user_id')
async def show_user(user_id):
    """Show detailed user profile"""
    profile = await face_rec.get_user_profile(user_id)
    
    if not profile:
        click.echo(f"User {user_id} not found")
        return
    
    import json
    click.echo(json.dumps(profile, indent=2))

@cli.command()
@click.argument('name')
@click.argument('image_path')
async def register(name, image_path):
    """Register new user from image"""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    result = await face_rec.register_user(name, image_bytes)
    
    if result['success']:
        click.echo(f" {result['message']}")
        click.echo(f"  User ID: {result['user_id']}")
    else:
        click.echo(f" {result['message']}")

@cli.command()
@click.argument('user_id')
@click.confirmation_option(prompt='Are you sure you want to delete this user?')
async def delete(user_id):
    """Delete user"""
    success = await face_rec.delete_user(user_id)
    
    if success:
        click.echo(f" User {user_id} deleted")
    else:
        click.echo(f" User {user_id} not found")

if __name__ == '__main__':
    cli(_anyio_backend="asyncio")
