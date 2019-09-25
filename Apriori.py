
import pandas as pd
def queryOption(option, df):
    template1 = "asso_rule.template1"
    template2 = "asso_rule.template2"
    template3 = "asso_rule.template3"
    print(option)
    if (option[:len(template1)] == template1):
        queryTemp1(option[len(template1) + 1: -1], df)
    elif(option[:len(template2)] == template2):
        queryTemp2(option[len(template2) + 1: -1], df)
    elif(option[:len(template3)] == template3):
        queryTemp3(option[len(template3) + 1: -1], df)
    return 
def queryTemp1(template1, df):
    result = pd.DataFrame(data=None, columns= df.columns)
    parts = eval(template1)
    total_set = set()
    if parts[0] == "RULE":
        if parts[1] == "ANY":
            for son in parts[2]:
                add_set = set()
                for line in range(len(df.RULE)):
                    print(str(line)+"\n")
                    print(type(df.RULE[line]))
                    #son_candidate = []
                    #son_candidate = parts[2].split(',')
                    father_candidate= []
                    father_candidate = df.RULE[line].split(',')
                    print("........."+str(set(father_candidate)))
                    #print(",,,,,,,,,"+str(set(son_candidate)))
                    if set([son]).issubset(set(father_candidate)):
                        print("!!!!!!!!!!!\n")
                        add_set.add(line)
                total_set = total_set.union(add_set)
            print(len(total_set),"rows selected")
        elif (parts[1] == "NONE"):
            total_set = set()
            for i in range(len(df.RULE)):
                total_set.add(i) 
            for son in parts[2]:
                minus_set = set()
                
                for line in range(len(df.RULE)):
                    print(str(line)+"\n")
                    print(type(df.RULE[line]))
                    #son_candidate = []
                    #son_candidate = parts[2].split(',')
                    father_candidate= []
                    father_candidate = df.RULE[line].split(',')
                    print("........."+str(set(father_candidate)))
                    #print(",,,,,,,,,"+str(set(son_candidate)))
                    if set([son]).issubset(set(father_candidate)):
                        print("!!!!!!!!!!!\n")
                        minus_set.add(line)
                total_set = total_set.difference(minus_set)
            print(len(total_set),"rows selected")
        elif (parts[1] == 1):
            for item in parts[2]:
                print(item)
    elif parts[0] == "HEAD":
        if parts[1] == "ANY":
            for son in parts[2]:
                add_set = set()
                for line in range(len(df.HEAD)):
                    print(str(line)+"\n")
                    print(type(df.HEAD[line]))
                    #son_candidate = []
                    #son_candidate = parts[2].split(',')
                    father_candidate= []
                    father_candidate = df.HEAD[line].split(',')
                    print("........."+str(set(father_candidate)))
                    #print(",,,,,,,,,"+str(set(son_candidate)))
                    if set([son]).issubset(set(father_candidate)):
                        print("!!!!!!!!!!!\n")
                        add_set.add(line)
                total_set = total_set.union(add_set)
            print(len(total_set),"rows selected")
        elif parts[1] == "NONE":
            total_set = set()
            for i in range(len(df.HEAD)):
                total_set.add(i) 
            for son in parts[2]:
                minus_set = set()
                for line in range(len(df.HEAD)):
                    print(str(line)+"\n")
                    print(type(df.HEAD[line]))
                    father_candidate= []
                    father_candidate = df.RULE[line].split(',')
                    print("........."+str(set(father_candidate)))
                    if set([son]).issubset(set(father_candidate)):
                        print("!!!!!!!!!!!\n")
                        minus_set.add(line)
                total_set = total_set.difference(minus_set)
            print(len(total_set),"rows selected")
        elif parts[1] == 1:
            print(parts[1])
    elif parts[0] == "BODY":
            if parts[1] == "ANY":
                for son in parts[2]:
                    add_set = set()
                    for line in range(len(df.BODY)):
                        print(str(line)+"\n")
                        print(type(df.BODY[line]))
                        #son_candidate = []
                        #son_candidate = parts[2].split(',')
                        father_candidate= []
                        father_candidate = df.BODY[line].split(',')
                        print("........."+str(set(father_candidate)))
                        #print(",,,,,,,,,"+str(set(son_candidate)))
                        if set([son]).issubset(set(father_candidate)):
                            print("!!!!!!!!!!!\n")
                            add_set.add(line)
                total_set = total_set.union(add_set)
                print(len(total_set),"rows selected")
            elif parts[1] == "NONE":
                total_set = set()
                for i in range(len(df.BODY)):
                    total_set.add(i) 
                for son in parts[2]:
                    minus_set = set()
                    for line in range(len(df.BODY)):
                        print(str(line)+"\n")
                        print(type(df.RULE[line]))
                        father_candidate= []
                        father_candidate = df.BODY[line].split(',')
                        print("........."+str(set(father_candidate)))
                        if set([son]).issubset(set(father_candidate)):
                            print("!!!!!!!!!!!\n")
                            minus_set.add(line)
                total_set = total_set.difference(minus_set)
                print(len(total_set),"rows selected")
    return
def queryTemp2(template2, df):
    parts = eval(template2)
    count = 0
    if parts[0] == "RULE":
        for line in range(len(df.RULE)):
            father_candidate= []
            father_candidate = df.RULE[line].split(',')
            if len(father_candidate) >= parts[1]:
                count += 1
        print(count)
    elif parts[0] == "HEAD":
         for line in range(len(df.HEAD)):
            father_candidate= []
            father_candidate = df.HEAD[line].split(',')
            if len(father_candidate) >= parts[1]:
                count += 1
         print(count)
    elif parts[0] == "BODY":
         for line in range(len(df.BODY)):
            father_candidate= []
            father_candidate = df.BODY[line].split(',')
            if len(father_candidate) >= parts[1]:
                count += 1
         print(count)
    return count
def queryTemp3(template3, df):
    parts = eval(template3)
    if parts[0] == "1or1":
        count1 = queryTemp1(str(parts[1:4]), df)
        count2 = queryTemp1(str(parts[4:]), df)
        print(max(count1, count2))
    elif parts[0] == "1and1":
        count1 = queryTemp1(str(parts[1:4]), df)
        count2 = queryTemp1(str(parts[4:]), df)
        print(min(count1, count2))
    elif parts[0] == "1or2":
        count1 = queryTemp1(str(parts[1:4]), df)
        count2 = queryTemp2(str(parts[4:]), df)
        print(max(count1, count2))
    elif parts[0] == "1and2":
        count1 = queryTemp1(str(parts[1:4]), df)
        count2 = queryTemp2(str(parts[4:]), df)
        print(min(count1, count2))
    elif parts[0] == "2or2":
        count1 = queryTemp2(str(parts[1:3]), df)
        count2 = queryTemp2(str(parts[3:]), df)
        print(max(count1, count2))
    elif parts[0] == "2and2":
        count1 = queryTemp2(str(parts[1:3]), df)
        count2 = queryTemp2(str(parts[3:]), df)
        print(min(count1, count2))
    return
a = "asso_rule.template1(\"BODY\", \"NONE\", ['G59_Up'])"
b = "asso_rule.template1(\"RULE\", 1, ['G59_UP', 'G10_Down'])"
c = "asso_rule.template2(\"RULE\", 3)"
d = "asso_rule.template3(\"2or2\", \"BODY\", 1, \"HEAD\", 2)"
df = pd.read_csv("Chart.csv")
#print(df.RULE)
queryOption(a, df)


