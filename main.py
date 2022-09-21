import os

tests = os.listdir('Tests')

for i in range(len(tests)):
    print(str(i) + '. ' + tests[i])

test_index = int(input("Index: "))

if test_index < 0 or test_index >= len(tests):
    print('The tests does not exist')
    exit()

exec(open("Tests/" + tests[test_index]).read())
