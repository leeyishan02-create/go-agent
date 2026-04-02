// Package main 定义了 go-agent 应用程序的主要包。
// tool.go 文件定义了工具（Tool）接口，所有可供 AI 代理调用的工具都必须实现该接口。
// 工具系统允许 AI 代理与外部系统交互，如读写文件、搜索网络等。
package main

import "encoding/json"

// Tool 接口定义了所有 AI 代理工具必须实现的方法。
// 每个工具都可以被 AI 模型在对话中自动调用，以完成特定任务。
// 工具的定义会被转换为 OpenAI 兼容的 function calling 格式发送给模型。
type Tool interface {
	// Name 返回工具的唯一名称标识符（如 "read"、"write"、"search"）。
	// 该名称用于在 API 请求中标识工具，以及在模型返回的 tool_call 中匹配对应的工具。
	Name() string

	// Description 返回工具的功能描述。
	// 该描述会作为 function calling 的 description 字段发送给模型，
	// 帮助模型理解何时以及如何使用该工具。
	Description() string

	// InputSchema 返回工具输入参数的 JSON Schema 定义。
	// 返回值是一个 map，通常包含 "type"、"properties" 和 "required" 等字段，
	// 用于告知模型该工具需要哪些参数以及参数的类型和描述。
	InputSchema() map[string]interface{}

	// Execute 执行工具的具体操作。
	// 参数 input 是 JSON 格式的原始消息（json.RawMessage 类型），包含模型传入的参数。
	// 返回值：
	//   - string：工具执行的结果（文本形式），会被反馈给模型作为上下文。
	//   - error：如果执行过程中发生错误则返回错误信息。
	Execute(input json.RawMessage) (string, error)
}
