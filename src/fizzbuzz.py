def fizzbuzz(n: int) -> list[str]:
    if n <= 0:
        return []
    
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    
    return result

# Test cases
def test_fizzbuzz():
    assert fizzbuzz(0) == []
    assert fizzbuzz(-1) == []
    assert fizzbuzz(1) == ["1"]
    assert fizzbuzz(2) == ["1", "2"]
    assert fizzbuzz(3) == ["1", "2", "Fizz"]
    assert fizzbuzz(4) == ["1", "2", "Fizz", "4"]
    assert fizzbuzz(5) == ["1", "2", "Fizz", "4", "Buzz"]
    assert fizzbuzz(6) == ["1", "2", "Fizz", "4", "Buzz", "Fizz"]
    assert fizzbuzz(15) == ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]

# Run tests
test_fizzbuzz()