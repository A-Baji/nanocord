import os

from openai import OpenAI, PermissionDeniedError
from datetime import datetime, timezone


def convert_timestamp(time: int):
    timestamp = datetime.fromtimestamp(time, tz=timezone.utc)
    human_readable_time = timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return human_readable_time


def convert_in_place(obj, key: str):
    obj[key] = convert_timestamp(obj[key])
    return obj


def list_jobs(openai_key: str = os.getenv("OPENAI_API_KEY"), full=False) -> list[dict]:
    client = OpenAI(api_key=openai_key)
    finetunes = client.fine_tuning.jobs.list()
    client.close()
    if full:
        return [
            convert_in_place(
                (
                    convert_in_place(j.model_dump(), "finished_at")
                    if j.finished_at
                    else j.model_dump()
                ),
                "created_at",
            )
            for j in finetunes.data
        ]
    else:
        return [
            {
                "id": j.id,
                "model": j.model,
                "status": j.status,
                "created_at": convert_timestamp(j.created_at),
                "finished_at": (
                    convert_timestamp(j.finished_at) if j.finished_at else None
                ),
            }
            for j in finetunes.data
        ]


def list_models(
    openai_key: str = os.getenv("OPENAI_API_KEY"), full=False
) -> list[dict]:
    client = OpenAI(api_key=openai_key)
    finetunes = client.models.list()
    client.close()
    if full:
        return [convert_in_place(f.model_dump(), "created") for f in finetunes.data]
    else:
        return [
            {"id": m.id, "created": convert_timestamp(m.created)}
            for m in finetunes.data
        ]


def get_job_info(job_id: str, openai_key: str = os.getenv("OPENAI_API_KEY")) -> dict:
    client = OpenAI(api_key=openai_key)
    job = client.fine_tuning.jobs.retrieve(job_id)
    client.close()
    return convert_in_place(
        (
            convert_in_place(job.model_dump(), "finished_at")
            if job.finished_at
            else job.model_dump()
        ),
        "created_at",
    )


def get_job_events(
    job_id: str, openai_key: str = os.getenv("OPENAI_API_KEY")
) -> list[dict]:
    client = OpenAI(api_key=openai_key)
    events = client.fine_tuning.jobs.list_events(job_id).data
    client.close()
    return [
        convert_in_place(
            j.model_dump(),
            "created_at",
        )
        for j in events
    ]


def cancel_job(job_id: str, openai_key: str = os.getenv("OPENAI_API_KEY")) -> dict:
    client = OpenAI(api_key=openai_key)
    client.fine_tuning.jobs.cancel(job_id)
    client.close()
    return {"result": f"Canceled fine-tuning job: {job_id}"}


def delete_model(
    model_name: str, openai_key: str = os.getenv("OPENAI_API_KEY"), force=False
) -> dict:
    confirm = (
        "yes"
        if force
        else input(
            "Are you sure you want to delete this model? This action is not reversable. Y/N: "
        )
    )
    if confirm.lower() not in ["y", "yes"]:
        print("Cancelling model deletion...")
        return
    client = OpenAI(api_key=openai_key)
    try:
        deleted = client.models.delete(model_name).model_dump()
    except PermissionDeniedError:
        deleted = {
            "error": "You have insufficient permissions for this operation. Missing scopes: api.delete"
        }
    client.close()
    return deleted
