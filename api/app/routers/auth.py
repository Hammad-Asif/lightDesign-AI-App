from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import auth
from app.db import get_db
from app.models import User
from app.schemas import LoginRequest, RegisterRequest, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _normalise(email: str) -> str:
    return email.strip().lower()


@router.post("/register", response_model=UserOut, status_code=201)
def register(body: RegisterRequest, response: Response, db: Session = Depends(get_db)):
    user = User(
        email=_normalise(body.email),
        name=body.name.strip(),
        password_hash=auth.hash_password(body.password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "An account with that email already exists")
    auth.start_session(db, user, response)
    return user


@router.post("/login", response_model=UserOut)
def login(body: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == _normalise(body.email)))
    if not auth.verify_password(user, body.password) or not user.is_active:
        raise HTTPException(401, "Incorrect email or password")
    auth.start_session(db, user, response)
    return user


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    auth.end_session(db, request, response)


# Answers "who is signed in?". Nobody is a normal answer rather than an error, so a signed-out
# visitor opening the app doesn't get a 401 logged in the browser console.
@router.get("/me", response_model=UserOut | None)
def me(user: User | None = Depends(auth.optional_user)):
    return user
