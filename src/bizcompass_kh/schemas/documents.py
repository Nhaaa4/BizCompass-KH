from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

SourceType = Literal["pdf", "html"]


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    agency: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl
    topic: str = Field(min_length=1)
    category: str = Field(min_length=1)
    published_year: int | None = None
    official: bool
    type: SourceType
