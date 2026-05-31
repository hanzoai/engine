"""
MCP (Model Context Protocol) client usage with mistral.rs Python API.

Connects to an MCP server, auto-discovers tools, and makes them available
to the model during conversations.

Install the filesystem server first: npx @modelcontextprotocol/server-filesystem . -y
"""

import hanzo


def main():
    # Connect to a local filesystem MCP server
    mcp_config = hanzo.McpClientConfigPy(
        servers=[
            hanzo.McpServerConfigPy(
                id="filesystem",
                name="Filesystem Tools",
                source=hanzo.McpServerSourcePy.Process(
                    command="npx",
                    args=["@modelcontextprotocol/server-filesystem", "."],
                    work_dir=None,
                    env=None,
                ),
            )
        ]
    )

    # Other transport types are also supported:
    #
    # McpServerSourcePy.Http(url="https://hf.co/mcp", timeout_secs=30, headers=None)
    # McpServerSourcePy.WebSocket(url="wss://api.example.com/mcp", timeout_secs=30, headers=None)
    #
    # For authentication, set bearer_token="your-token".
    # To avoid tool name conflicts, set tool_prefix="prefix".

    runner = hanzo.Runner(
        which=hanzo.Which.Plain(
            model_id="Qwen/Qwen3-4B",
            arch=hanzo.Architecture.Qwen3,
        ),
        mcp_client_config=mcp_config,
    )

    request = hanzo.ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "List the files in the current directory."}
        ],
        max_tokens=1000,
        tool_choice=hanzo.ToolChoice.Auto,
    )

    response = runner.send_chat_completion_request(request)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
