while True:
    number = int(input("how many lesson do you have this term: "))
    grade = 0
    total = 0
    unitT = 0
    for i in range(number):
        grade += float(input(f"grade {i+1}:"))
        unit = int(input("how many unit is that lesson:"))
        total += grade * unit
        unitT += unit
        grade = 0

    final = total/unitT
    print("your average is: ", final)
