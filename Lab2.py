'''
Lab 2
'''
print ("Lab 2")

##########Part 1 ###########
print("\n- - - - - Part 1 - - - - -")

'''
    1) Use a while loop to print all integers from 0 to 10, including 0 and 10.
'''
# YOUR CODE GOES HERE
print("\n- - - 1:") 
integer = 0
while (integer <= 10):
    print(integer)
    integer += 1

'''
    2) Use a while loop to print all even numbers from 2 up to and including 12
'''
# YOUR CODE GOES HERE
print("\n- - - 2:")
integer = 2
while integer <= 12:
    if integer%2 == 0:
        print(integer)
    integer += 1

'''
    3) Use a while in loop to ask the user for a number and 
    repeats this request if the number is not greater than 0
'''
# YOUR CODE GOES HERE
print("\n- - - 3:")
number = float(-1)
while number < 0:
    print("Enter a number ( Enter a number less than 0 to STOP ):\n")
    number = float(input("Number: "))
    print("Input: ",number)



######### Part 2 ###########
print("\n- - - - - Part 2 - - - - -")

'''
    1) Use a For loop to calculate -1000-999-...+-1+0+1+2+3+4+......+1000 
'''
# YOUR CODE GOES HERE
print("\n- - - 1:")
sum = int(0)
for e in range(-1000, 1001):
    sum += e
print(sum)
'''
    2) Use a For loop to print all positive odd numbers less than 8, except 5.
'''
# YOUR CODE GOES HERE
print("\n- - - 2:")
for e in range(1,8):
    if e%2 == 0:
        doNothing = 0
    else:
        if e != 5:
            print(e)


'''
    3) Use a For loop to calculate the average of 1,2,3,4,......,1000 
'''
# YOUR CODE GOES HERE
print("\n- - - 3:")
sum = 0
start = 1
end = 1001

for e in range(start,end):
    sum+=e

sum /= 1000

print(sum)

######### Part 3 ###########
print("\n- - - - - Part 3 - - - - -")
'''
    1) write a code to print all the elements in a list
    e.g.:  [1, "Fine", 3, 18, -5.1] --> 1
                                        "Fine"
                                        3
                                        18
                                        -5.1
'''
# YOUR CODE GOES HERE
print("\n- - - 1:")
list = [1, "Fine", 3, 18, -5.1]
for e in list:
    print(e)

'''
    2) write a code to find the largest number in the list of numbers and its index 
    e.g.:  [1, 0, 3, 18, -5] --> 18 , 3
'''
# YOUR CODE GOES HERE
print("\n- - - 2:")
list = [1, 0, 3, -5, 19]
index = 0
number = list[0]
for e in range(len(list)):
    if number < list[e]:
        number = list[e]
        index = e
print("List: ", list)
print("Largest Number:", number, "Index:", index)

'''
    3) return the average of numbers in the list excluding the last element
    e.g.:  [1, 0, 3, 5, -9, 2]--> 0
    '''
# YOUR CODE GOES HERE
print("\n- - - 3:")
list = [1, 0, 3, 5, -9, 2]
avg = 0
for e in range(len(list) -1 ):
    avg += list[e]

print("List:",list)
print("avg:", ( avg/( len(list)-1 )) )    
    
######### Part 4 ###########
print("\n- - - - - Part 4 - - - - -")
'''
    1) given a string, return the reversed of that
    e.g.:  "Hello" --> "olleH"
'''
# YOUR CODE GOES HERE
print("\n- - - 1:")
string = "Hello"
newString = ""
for e in string:
    newString = e + newString

print(newString)


'''
    2) 10.	given a string, return the count of 'f' letter in it
    e.g.:  "fatherflytodayorfridayfreedom" --> 4
'''
# YOUR CODE GOES HERE
print("\n- - - 2:")
string = "fatherflytodayorfridayfreedom"
count = 0
for e in string:
    if e == "f":
        count+=1
print("string: ",string)
print("# of f's: ", count)

'''
    3) given a string, return the same string removing all the 'f' letters in it
    e.g.:  "fatherflytodayorfridayfreedom" --> "atherlytodayorridayreedom"
'''
# YOUR CODE GOES HERE
import re
print("\n- - - 3:")
string = "fatherflytodayorfridayfreedom"
print("BEFORE: String: ", string)
line = re.sub('f', '', string)
print("AFTER: String: ", line)


######### Part 5 ###########
print("\n- - - - - Part 5 - - - - -")
"""
    1) Write and test the following function:
		Identifier: Addition
		Args:
			- two float numbers
		Returns:
			- summation of the two numbers
"""
# YOUR CODE GOES HERE
print("\n- - - 1:")
def Addition(f1, f2):
    return(f1+f2)
print("Addition of 2 & 2:",Addition(2,2))

"""    
	2) Write and test the following function:
		Identifier: fahren_to_celcius
		Args:
			- float temp_fahren
		Returns:
			- temp_fahren converted to Celcius

		Note: C = (F-32) * (5/9)
"""
# YOUR CODE GOES HERE
print("\n- - - 2:")
def fahren_to_celcius(f):
    return( (float(f) - 32) * (5/9) )
print("Faren of 32 in Celcius", fahren_to_celcius(32))

"""
	3) Write and test a function which:
		Identifier: Binary_Addition
		Args:
			- two strings: e.g. '100','1'
		Returns:
			- string: summation of the two numbers : e.g. '101'
"""
# YOUR CODE GOES HERE
print("\n- - - 3:")
def Binary_Addition(str1, str2):
    return bin(int(str1,2) + int(str2,2))

print("Binary Addition of '100' & '1' is", Binary_Addition("100","1"))
print("Binary Addition of '11' & '11' is", Binary_Addition("11","11"))

# blank prints are just for space-ers
print() 