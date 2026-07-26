"""Shared AWS session builder — a typo'd profile fails loudly, never silently.

Falling back to the default credential chain when aws.profile doesn't exist
would let a typo read from (or bill against) a DIFFERENT account, so every
boto3 session in the library is built through here instead.
"""
import boto3
from botocore.exceptions import ProfileNotFound


def aws_session(profile: str, region: str) -> boto3.Session:
    """Session for config.yaml's aws.profile; empty profile → default chain."""
    profile = (profile or "").strip()
    try:
        return (
            boto3.Session(profile_name=profile, region_name=region)
            if profile
            else boto3.Session(region_name=region)
        )
    except ProfileNotFound:
        available = boto3.Session().available_profiles
        raise RuntimeError(
            f"AWS profile {profile!r} (config.yaml aws.profile) not found in "
            f"~/.aws — available profiles: {available or 'none'}. Fix the name, "
            f"or leave it empty to use the default credential chain."
        ) from None
