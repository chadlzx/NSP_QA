#使用json快速保存各类cache
#目前仅提供savejson loadjson

import json
import re
import time
from drop_eval import get_metrics

from prettytable import PrettyTable

def handle_span_spans(e1,e2):
    a,b,c = e1
    d,e,f = e2
    a = a*c + d*f
    b = b*c + e*f
    c = c+f
    if c!=0:
        return [a/c, b/c, c]
    else :
        return [0.0,0.0,0]

def clear_number(e1):
    a,b,c = e1
    return [round(a,4),round(b,4),round(c)]

def print_eval_result(answering_abilities, eval_result):
    types = ["number","span","spans" ,"span+spans","date",]
    if "use way:best_answer actual type:number" in eval_result:
        answering_abilities = answering_abilities + ['best_answer',]

    for way in answering_abilities:
        key0 = "use way:{} actual type:{}".format(way,"span")
        key1 = "use way:{} actual type:{}".format(way,"spans")
        key2 = "use way:{} actual type:{}".format(way,"span+spans")
        eval_result[key2] = handle_span_spans(eval_result[key0], eval_result[key1])
    if "span+spans" not in eval_result:
        eval_result["span+spans"] = handle_span_spans(eval_result["span"], eval_result["spans"])

    table = PrettyTable(["way"] + types + ["real"])
    for way in answering_abilities:
        row = [way,]
        for type_ in types:
            key = "use way:{} actual type:{}".format(way,type_)
            row.append(str(clear_number(eval_result[key])))
        
        key = "real:{}".format(way)
        row.append(str(clear_number(eval_result[key])))
        
        table.add_row(row)


    row = ["total",]
    for type_ in types:
        key = type_
        row.append(str(clear_number(eval_result[key])))
    e_total = ["total_em","total_f1","total_count"]
    e_total = [ eval_result[i] for i in e_total]
    row.append(str(clear_number(e_total)))
    table.add_row(row)

    return table.get_string()

#
def savejson(file_path, file):
    with open(file_path, mode='w', encoding='utf-8') as f:
        f.write(json.dumps(file, indent=4))
def loadjson(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        reader = json.loads(f.read())
    return reader

def getNumberFromString(string):
    if isinstance(string, float) or isinstance(string, int):
        return [string,]
    numberList = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", string)
    for idx, num in enumerate(numberList):
        try:
            num = abs(float(re.sub(',','',num)))
        except:
            num = 0.0
        numberList[idx] = num
    if len(numberList) == 0:
        numberList = [0.0,]
    return numberList


def add_string(text):
    if text is None:
        text = ""
    if "\"" in text:
        text = text.replace("\"","\"\"")
    return "\"" +text + "\""


def drop_eval_check_answer(answers, result):
    max_f1, max_em = 0.0, 0.0
    for answer in answers:
        em_score, f1_score = get_metrics(result, answer)
        max_f1 = max(max_f1, f1_score)
        max_em = max(max_em, em_score)
    return max_em , max_f1

#not used
def checkAnswer(answer, result):
    if answer == None or result == None:
        return False
    answer = answer.lower()
    answer = answer.strip()
    result = result.lower()
    result = result.strip()
    if answer == "" or result == "":
        return False
    if answer == result:
        return True
    numberAnswer = getNumberFromString(answer)
    numberResult = getNumberFromString(result)
    if numberAnswer != [0,] and numberResult != [0,]:
        numberAnswer.sort()
        numberResult.sort()
        if numberAnswer == numberResult:
            return True
    if numberAnswer == [0.0,] and numberResult == [0.0,]:
        answer = answer.split(' ')
        result = result.split(' ')
        flagAnswer = False
        for ans in answer:
            if ans not in result:
                flagAnswer = True
        if flagAnswer == False:
            return True

        flagAnswer = False
        for ans in result:
            if ans not in answer:
                flagAnswer = True
        if flagAnswer == False:
            return True
        
    return False


def answer_text(answer_dict):
    if answer_dict == None:
        return "",""
    number = answer_dict['number'].strip()
    if number:
        return number, 'number'
    
    spans = answer_dict['spans']
    spans_str = " ".join([span.strip() for span in spans if span.strip()])
    if spans_str:
        if len(spans) > 1:
            return spans_str,'spans'
        else:
            return spans_str,'span'
    
    date = answer_dict['date']
    if len(date) != 3:
        return "",""
    date = ' '.join([d.strip() for d in [date['day'], date['month'], date['year']] if d.strip()])
    if date:
        return date,'date'
    return "",""



