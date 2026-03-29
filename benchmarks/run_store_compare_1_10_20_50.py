import run_store_compare as base
from run_store_compare import Scenario

def build_scenarios_override():
    sizes = [1, 10, 20, 50]
    return [Scenario("single", mb, 1) for mb in sizes] + [Scenario("multi", n, n) for n in sizes]

base.build_scenarios = build_scenarios_override

if __name__ == "__main__":
    base.main()
