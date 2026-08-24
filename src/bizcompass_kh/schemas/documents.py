from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

SourceType = Literal["pdf", "html"]

class SourceConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    
    id: str
    agency: str
    title: str
    url: str
    topic: str
    category: str
    published_year: int | None
    official: bool
    type: SourceType