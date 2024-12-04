from pydantic import BaseModel, Field


# Classes and functions from the original script
class CompanyName(BaseModel):
    company_name: str = Field(
        description="Company name extraction from the query",
        pattern=r"""^\w[\w.\-#&\s]*$""",
    )


class Date(BaseModel):
    date: str = Field(
        description="A date of format `YYYY-MM-DD, valid from the 21st century",
        pattern=r"^(?:20)\d\d-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$",
    )


class Ticker(BaseModel):
    company_symbol: str = Field(
        description="Company symbol from NSE/BSE, should only contain capital letters.",
        pattern="^[A-Z]{1,12}$",
    )
