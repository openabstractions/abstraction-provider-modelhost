package service

import "testing"

func TestEngineChecksPreserveCapturedObservation(t *testing.T) {
	check := EngineCheck{Kind: EngineCheckKind, Copy: "release", State: EngineCurrent, Checked: "2026-11-02T00:00:00.000000Z"}
	read := EngineChecks(check)
	first := read()
	if len(first) != 1 || first[0] != check {
		t.Fatal(first)
	}
	first[0].State = EngineRefused
	if again := read(); len(again) != 1 || again[0] != check {
		t.Fatal(again)
	}
}
