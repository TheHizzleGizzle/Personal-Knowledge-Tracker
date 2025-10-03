from fastapi import APIRouter

router = APIRouter()


@router.get("/", summary="API health probe")
def read_health() -> dict[str, str]:
    return {"status": "ok"}