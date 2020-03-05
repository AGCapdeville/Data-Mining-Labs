'''
Lab 1
'''
print ("Lab 1")
##########Part 0 ###########

'''
   1) Create a variable, x, and set its value to 5
      Create a variable, y, and set its value to 3 more than x's value
      Exponentiate x by half of y, and store the result in x (hint: **=)
'''
# YOUR CODE GOES HERE
print()
print("= = = = = = = = = = = = Part 0.1 = = = = = = = = = = = = ")
x = 5
y = x+3
print("X:",x)
print("Y:",y)
x = x ** (y/2)
print("( After X**Y/2 ):")
print("X:",x)



'''
    2)  A rectangle has the width and height defined below.
        Store its area in a new variable.
        Print out your result and check your answer.
        Change the values of width and height and ensure the result is still correct.
'''
print()
print("= = = = = = = = = = = = Part 0.2 = = = = = = = = = = = = ")
width = 3
height = 4
# YOUR CODE GOES HERE
area = width * height
print("width:",width,"height:",height,"area:",area)
print("new values: width:5 height:5")
width = 5
height = 5
area = width * height
print("new area:",area)





######### Part 1 ###########
print()
print("= = = = = = = = = = = = Part 1.1 = = = = = = = = = = = = ")

x = 4
y = -7
z = 10

print(x==4)
print(x>4)
print(x<=y)
print(x>=z-11)
print(x!=z-8%11)
print((2*z+1)%11>3*x-1)



######### Part 2 ###########

'''
    1) Use an if-else sequence to print whether x is even or odd.
       HINT: use the modulus operator %
'''
x = 12
# YOUR CODE GOES HERE
print()
print("= = = = = = = = = = = = Part 2.1 = = = = = = = = = = = = ")
if x%2 == 0:
    print("+ even")
else:
    print("- odd")


'''
	2) Write a program which:
		
		- stores their two inputs in variables as floats (e.g. num1 and num2)
		- stores an arithmetic operation ( + - * / %) as a string (op = '+')
		- stores the result of the specfied operation with the specified numbers
			- if the user's input operation isn't one of those listed above, instead print a message telling them so
			- if the inputs are invalid for the operation (i.e. dividing by 0) print a message informing the user
		- if the inputs were valid, prints out the resulting equation
'''
# num1 = 2.1
# num2 = 3
# op = '*'

# YOUR CODE GOES HERE
print()
print("= = = = = = = = = = = = Part 2.2 = = = = = = = = = = = = ")


def getNum(s):
    while True:
        print("    Number",s)
        number = input("Enter number :")
        try:
            number = float(number)
            if isinstance(number, float) or isinstance(number, int):
                return number
        except:
            print("        INVALID NUMBER")
    
def getOp():
    while True:
        op = input("Enter an operator ( + - * / % ) :")
        if op=="+" or op=="-" or op=="*" or op=="/" or op=="%":
            return op
        print("    INVALID OP")

def tryOp(n1, n2, op):
    if op=="+":
        print(n1 + n2)
        return True
    if op=="-":
        print(n1 + n2)
        return True
    if op=="*":
        print(n1 * n2)
        return True
    if op=="/":
        if n2==0:
            return False
        else:
            print(n1 / n2)
            return True
    if op=="%":
        print(n1 % n2)
        return True

while True:
    n1 = getNum(1)
    n2 = getNum(2)
    op = getOp()
    if tryOp(n1, n2, op):
        cont = input("Test another input? ( Enter yes / no ): ")
        if cont=="no":
            break
    else:
        print("ERROR, INVALID OPERATION")



######### Part 3 ###########

'''
    1) Use a while loop to print all integers from 0 to 10, including 0 and 10.
'''
# YOUR CODE GOES HERE
print()
print("= = = = = = = = = = = = Part 3.1 = = = = = = = = = = = = ")
i=0
while i < 11:
    print(i)
    i+=1

'''
    2) Use a while loop to print all even numbers from 2 up to and including 12
'''
# YOUR CODE GOES HERE
print()
print("= = = = = = = = = = = = Part 3.2 = = = = = = = = = = = = ")
i=2
while i <= 12:
    if i%2 == 0:
        print(i)
    i+=1

######### Part 4 ###########


'''
    1) Use a For loop to calculate -1000-999-...+-1+0+1+2+3+4+......+1000 

'''
# YOUR CODE GOES HERE
print()
print("= = = = = = = = = = = = Part 4.1 = = = = = = = = = = = = ")
ans = 0
for i in (-1000, 1000): 
    ans += i
print(ans)

'''
    2) Use a For loop to print all positive odd numbers less than 8, except 5.
'''
# YOUR CODE GOES HERE
print()
print("= = = = = = = = = = = = Part 4.2 = = = = = = = = = = = = ")

for x in range(8):
    if x%2 == 0:
        DoNothing=0
    elif x != 5:
        print(x)


