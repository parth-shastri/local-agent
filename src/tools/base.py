from pydantic import BaseModel
from typing import Optional, Type
from typing import Any, Callable
from inspect import signature
from pydantic import create_model
from pydantic.fields import FieldInfo


def _create_function_schema(fn: Callable, name: str) -> Type[BaseModel]:
    """Create a function schema from the fn"""
    fields = {}
    params = signature(fn).parameters
    for param_name in params:
        param_type = params[param_name].annotation
        param_default = params[param_name].default

        if param_type is params[param_name].empty:
            param_type = Any

        if param_default is params[param_name].empty:
            # Required field
            fields[param_name] = (param_type, FieldInfo())
        elif isinstance(param_default, FieldInfo):
            # Field with pydantic.Field as default value
            fields[param_name] = (param_type, param_default)
        else:
            fields[param_name] = (param_type, FieldInfo(default=param_default))

    return create_model(name, **fields)


class Tool(BaseModel):
    func: Callable
    tool_name: str
    tool_description: str
    tool_schema: Type[BaseModel]
    return_direct: bool = False

    def tool_output(self, input: Any):
        """Call the tool"""
        return self.func(**input)

    @classmethod
    def from_function(
        cls,
        function: Callable,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        function_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = True,
    ) -> "Tool":
        """simply pass in a function to extract the tool_desc and the schema"""
        # Extract tool name and description from a input function.
        tool_name = tool_name or function.__name__
        docstring = function.__doc__
        # create the desc
        tool_desc = tool_description or f"{tool_name}{signature(function)}\n{docstring}"

        # define the function schema
        tool_schema = function_schema or _create_function_schema(function, tool_name)

        return cls(
            func=function,
            tool_name=tool_name,
            tool_description=tool_desc,
            tool_schema=tool_schema,
            return_direct=return_direct,
        )

    def get_parameter_dict(self) -> dict:
        """Get Parameter dict from the schema"""
        parameters = self.tool_schema.model_json_schema()
        parameters = {
            k: v
            for k, v in parameters.items()
            if k in ["type", "properties", "required", "definitions", "$defs"]
        }
        return parameters

    def to_openai_tool(self, skip_length_check: bool = False):
        """Convert to openai tool calling format (Apparently thats what Ollama takes in as well.)"""
        if not skip_length_check and len(self.tool_description) > 1024:
            raise ValueError(
                "Tool description exceeds the length of maximum characters. "
                "Please shorten your description or move it to prompt"
            )

        tool = {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description,
                "parameters": self.get_parameter_dict(),
            },
        }

        return tool
