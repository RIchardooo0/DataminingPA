def queryTemp1(template1):
    parts = eval(template1)
    if parts[0] == "RULE":
        print(parts[0])
        if parts[1] == "ANY":
            print(parts[1])
        elif parts[1] == "NONE":
            print(parts[1])
        elif parts[1] == 1:
            print(parts[1])
    elif parts[0] == "HEAD":
        print(parts[0])
        if parts[1] == "ANY":
            print(parts[1])
        elif parts[1] == "NONE":
            print(parts[1])
        elif parts[1] == 1:
            print(parts[1])
    elif parents[0] == "BODY":
        print(parts[0])
        if parts[1] == "ANY":
            print(parts[1])
        elif parts[1] == "NONE":
            print(parts[1])
        elif parts[1] == 1:
            print(parts[1])
    return
def queryTemp2(template2):
    parts = eval(template2)
    if parts[0] == "RULE":
        print(parts[0])
    elif parts[0] == "HEAD":
        print(parts[0])
    elif parents[0] == "BODY":
        print(parts[0])
    return
def queryTemp3(template3):
    parts = eval(template3)
    if parts[0] == "1or1":
        print(parts[0])
    elif parts[0] == "1and1":
        print(parts[0])
    elif parts[0] == "1or2":
        print(parts[0])
    elif parts[0] == "1and2":
        print(parts[0])
    elif parts[0] == "2or2":
        print(parts[0])
    elif parts[0] == "2and2":
        print(parts[0])
    return
a = "asso_rule.template1(\"RULE\", \"ANY\", ['G59_UP', 'G10_Down'])"
b = "asso_rule.template1(\"RULE\", 1, ['G59_UP', 'G10_Down'])"
queryOption(b)
