# Run Tests

Run the full pytest test suite and report results.

## Steps

1. Run `pytest tests/ -v --tb=short`
2. If failures exist, show which nodes or retrievers are failing
3. Do NOT fix failing tests by weakening assertions — fix the source code
4. After fixing, re-run to confirm all pass

## Expected Pass Conditions

- test_retrievers.py: all metric tests pass without KB (mocked)
- test_agent_nodes.py: all node tests pass without KB (mocked)

## If Tests Fail After a Code Change

Check: did you change a function signature or return format?
The tests verify the interface contract. If you changed the contract
intentionally, update the test to match AND update ARCHITECTURE.md.
