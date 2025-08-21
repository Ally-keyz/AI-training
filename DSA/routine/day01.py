
# implementing the fibonacci numbers algorithms using for loops
prev2 = 0
prev1 = 1
numbers = []
numbers.append(0)
for num in range(18):
    newNum = prev1 + prev2
    numbers.append(newNum)
    prev2 = prev1
    prev1 = newNum


print("Numbers :",numbers)
    


# implementation of the same process using recursion

count = 2
fibonacci_numbers = []
fibonacci_numbers.append(0)
def fibo_recursion(prev01,prev02):
    global count
    if count <= 19:
        newNum = prev01 + prev02
        fibonacci_numbers.append(newNum)
        prev02 = prev01
        prev01 = newNum
        count += 1
        fibo_recursion(prev01 , prev02)
    else:
        return


fibo_recursion(1,0)
print("Numbers :",fibonacci_numbers)
        
