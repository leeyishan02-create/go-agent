package main

import "encoding/json"

type Tool interface {
	Name() string
	Description() string
	InputSchema() map[string]interface{}
	Execute(input json.RawMessage) (string, error)
}
