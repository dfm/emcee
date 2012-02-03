from sampler import *
from mh import *
from ensemble import *

def test():
    from tests import Tests

    test_names = [
                  ("Parallel Sampler",    "test_parallel"),
                  ("Ensemble Sampler",    "test_ensemble"),
                  ("Metropolis-Hastings", "test_mh"),
                 ]

    print "Starting tests..."

    failures = 0
    tests = Tests()
    for t in test_names:
        tests.setUp()
        try:
            getattr(tests, t[1])()
        except:
            print "Test: %s failed."%(t[0])
            failures += 1
        else:
            print "Test: %s passed."%(t[0])

    print "%d tests passed"%(len(test_names)-failures)
    print "%d tests failed"%(failures)

