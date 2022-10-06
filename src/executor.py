
import number_type
import numpy as np
import logging
logger = logging.getLogger()
import traceback
import re

class node(object):
    def __init__(self, fa=None, token="",type="") -> None:
        super().__init__()
        self.fa = fa
        self.child = []
        self.token = token
        self.type = type
        self.depth = 0
        self.answer = None
        self.real_program = None
    def getfa(self):
        return self.fa
    def getchild(self, index):
        return self.child[index]

def get_next_double_quote(string, start_idx):
    idx = start_idx
    if string[idx+1] =='\"':
        raise Exception("invalid string!ERROR HAPPENED IN get_next_double_quote! Help!")
    idx += 1
    while idx < len(string):
        if string[idx] =="\"" and (idx+1 ==len(string) or string[idx+1]!="\"" ):
            return idx
        idx += 1
    raise Exception("Can't find useful \"")

def get_word(string, start_idx):
    idx = start_idx
    while idx < len(string):
        s = string[idx]
        if (s > 'z' or s < 'a') and (s > 'Z' or s < 'A'):
            break
        idx += 1
    return string[start_idx: idx], idx

def get_word2(string, start_idx):
    idx = start_idx+1
    while idx < len(string):
        s = string[idx]
        if (s > '9' or s < '0') and (s !='_'):
            break
        idx += 1
    return string[start_idx: idx], idx

def get_word3(string, start_idx):
    idx = start_idx+1
    while idx < len(string):
        s = string[idx]
        if (s > '9' or s < '0'):
            break
        idx += 1
    return string[start_idx:idx], idx
# parameter:
# passage: a {split}.coreNLP.json dict

month_table = ['January','February','March','April','May','June','July','August','September','October','November','December']


class executor(object):
    def __init__(self, program, questionId, reader,old2new_dict=None) -> None:
        super().__init__()
        if program == None:
            program = ""
        self.program = program
        self.rt = None
        self.reader = reader
        self.questionId = questionId
        self.number2token = reader.questionId2number2token(questionId)
        self.old2new_dict = old2new_dict
        self.max_depth = 0

        self.diff1flag = False
        if re.match("^diff\(1,[NQ][0-9]+\)$",self.program) or re.match("^diff\([NQ][0-9]+,1\)$",self.program):
            self.diff1flag = True
            #print(self.program)
        
    def get_clear_question(self):
        question = self.reader.questionId2corenlp_qa_pair(self.questionId)['question']
        clear_question = self.reader.clear_token(question)
        return clear_question
    def get_clear_passage(self):
        passageId = self.reader.questionId2passageId(self.questionId)
        clear_passage = self.reader.passageId2clear_passage(passageId)
        return clear_passage

    def tokenize(self, program: str):
        if program.strip() == "":
            raise Exception("No tokenized program!")
        _type = [] #str or number or operator
        _token = []
        idx = 0 
        while idx < len(program):
            s = program[idx]
            if s == " ":
                idx += 1
                continue
            elif s == '\"':
                t = get_next_double_quote(program, idx)
                _token.append(program[idx+1:t])
                _type.append("string")
                idx = t+1
            elif s == '(':
                _token.append(s)
                _type.append('leftpare')
                idx += 1
                continue
            elif s == ')':
                _token.append(s)
                _type.append('rightpare')
                idx += 1
                continue
            elif s == ',':
                _token.append(s)
                _type.append('comma')
                idx += 1
                continue
            elif s == 'N' or s=='X' or s=='Q':
                word,t = get_word2(program, idx)
                _token.append(word)
                _type.append('number')
                idx = t
                continue
            elif  '0' <= s <= '9':
                word,t = get_word3(program, idx)
                _token.append(word)
                _type.append('number')
                idx = t
                continue
            else :
                word,t = get_word(program, idx)
                if word not in ['kv','add','diff','argmin','argmax','min','max','count','year','month','day','hour','minute','second','mul','div','avg','spanq','spanp',"million",'billion','decade']:
                    raise Exception("Unknown Operator!")
                _token.append(word)
                _type.append('operator')
                idx = t
        
        return {'token':_token,'type':_type}
        
    def get_program_syntax_tree(self):
        if self.rt :
            return
        program = self.program
        if isinstance(program, str):
            program = self.tokenize(program)
        _token = program['token']
        _type = program['type']
        now = node(token="ROOT",type="ROOT")
        self.rt = now
        pre = ""
        for i in range(len(_token)):
            
            token= _token[i]
            type = _type[i]
            #print(token, type)
            if type == 'operator':
                if pre != "" and pre != 'leftpare' and pre != 'comma':
                    raise Exception("Error Expression")
                temp = node(fa=now,token=token,type=type)
                now.child.append(temp)
            elif type == "string":
                if pre != "" and pre != 'leftpare' and pre != "comma":
                    raise Exception("Error Expression")
                temp = node(fa=now,token=token,type=type)
                now.child.append(temp)
            elif type == "number":
                if pre != "" and pre != 'comma' and pre != 'leftpare':
                    raise Exception("Error Expression")
                temp = node(fa=now,token=token,type=type)
                now.child.append(temp)
            elif type == 'comma':
                if pre == 'comma':
                    raise Exception("Error Exception!")
                pass
            elif type == "leftpare":
                if pre != 'operator' and pre != 'comma' and pre != 'leftpare' :
                    raise Exception("Error Expression")

                if pre == 'comma' or pre == 'leftpare' :
                    temp = node(fa=now,token="key-value",type="operator")
                    now.child.append(temp)
                now = now.child[len(now.child)-1]
            elif type == "rightpare":
                #if self.rt == now:
                #    continue
                now = now.fa
            pre = type
        if self.rt is not now:
            raise Exception("Error Expression!")
    
    def compute_answer(self, n):
        if n.answer != None:
            return
        n.depth = 0
        for c in n.child:
            self.compute_answer(c)
            n.depth = max(n.depth, c.depth + 1)
        if n.type == 'operator':
            if n.token == 'add':
                n.answer = n.child[0].answer
                if n.child == []:
                    raise Exception("No Enough Parameter for Add")
                for c in n.child[1:]:
                    if isinstance(n.answer, list) or isinstance(c.answer, list):
                        raise Exception("Error Parameter for add")
                    n.answer = n.answer + c.answer
            if n.token == 'diff':
                if n.child!= [] and n.child[0] != None and n.child[1] != None and len(n.child) == 2:
                    n.answer =  n.child[0].answer - n.child[1].answer
                else :
                    raise Exception("Error Parameter for diff")
            if n.token == 'key-value' or n.token == 'kv':
                n.answer = []
                for c in n.child:
                    n.answer.append(c.answer)
            if n.token == 'min':
                if n.child == []:
                    raise Exception("Error Parameter for min")
                n.answer = n.child[0].answer
                
                for c in n.child[1:]:
                    n.answer = min(n.answer , c.answer)
            if n.token == 'max':
                if n.child == []:
                    raise Exception("Error Parameter for max")
                n.answer = n.child[0].answer
                for c in n.child:
                    n.answer = max(n.answer , c.answer)
            if n.token == 'argmax':
                if n.child == []:
                    raise Exception("Error Parameter for argmax")
                n.answer, Max =  n.child[0].answer[0], n.child[0].answer[1]
                for c in n.child:
                    values = c.answer[1:]
                    for value in values:
                        if Max < value:
                            n.answer = c.answer[0]
                            Max = value
            if n.token == 'argmin':
                if n.child == []:
                    raise Exception("Error Parameter for argmin")
                n.answer, Min =  n.child[0].answer[0], n.child[0].answer[1]
                for c in n.child:
                    values = c.answer[1:]
                    for value in values:
                        if Min > value:
                            n.answer = c.answer[0]
                            Min = value
            if n.token == 'count':
                n.answer = number_type.number(len(n.child))
            if n.token == 'mul':
                if n.child == []:
                    raise Exception("Error Parameter for mul")
                n.answer = n.child[0].answer
                for c in n.child[1:]:
                    n.answer = n.answer * c.answer
            if n.token == 'div':
                if n.child == [] or len(n.child) != 2:
                    raise Exception("Error Parameter for max")
                n.answer = n.child[0].answer/n.child[1].answer
            if n.token == 'avg':
                if n.child == []:
                    raise Exception("Error Parameter for mul")
                n.answer = n.child[0].answer
                for c in n.child[1:]:
                    n.answer = n.answer + c.answer
                n.answer = n.answer / number_type.number( len(n.child) )
            if n.token == 'year':
                if isinstance(n.child[0].answer, number_type.myDate)  :
                    n.answer = number_type.number(n.child[0].answer.date['year'])
                if isinstance(n.child[0].answer, number_type.deltaDate):
                    n.answer = number_type.number(n.child[0].answer.to_year())
                else:
                    n.answer = n.child[0].answer
            if n.token == 'month':
                if isinstance(n.child[0].answer, number_type.myDate) :
                    ind = number_type.number(n.child[0].answer.date['month']).value
                    ind = int(ind)
                    n.answer = month_table[ind - 1]
                if isinstance(n.child[0].answer, number_type.deltaDate):
                    n.answer = number_type.number(n.child[0].answer.to_month())
                else:
                    n.answer = n.child[0].answer
            if n.token == 'day':
                if isinstance(n.child[0].answer, number_type.myDate) or isinstance(n.child[0].answer, number_type.deltaDate):
                    n.answer = number_type.number(n.child[0].answer.date['day'])
                else:
                    n.answer = n.child[0].answer
            if n.token == 'hour':
                if isinstance(n.child[0].answer, number_type.time):
                    n.answer = number_type.number(n.child[0].answer.date[0])
                else:
                    n.answer = n.child[0].answer
            if n.token == 'minute':
                if isinstance(n.child[0].answer, number_type.time):
                    n.answer = number_type.number(n.child[0].answer.date[1])
                else:
                    n.answer = n.child[0].answer
            if n.token == 'second':
                if isinstance(n.child[0].answer, number_type.time):
                    n.answer = number_type.number(n.child[0].answer.date[1]*60)
                else:
                    n.answer = n.child[0].answer
            
            #just for execute if_run, not handle
            if n.token == 'million':
                n.answer = n.child[0].answer
            if n.token == 'billion':
                n.answer = n.child[0].answer
            if n.token == 'decade':
                n.answer = n.child[0].answer



            if n.token == 'spanp':
                start_pos = round(n.child[0].answer.value)
                end_pos =   round(n.child[1].answer.value)
                passage = self.get_clear_passage()
                passage = passage.strip().split(' ')
                passage = [word.strip() for word in passage if word!='']
                passage = passage[start_pos: end_pos+1]
                n.answer = ""
                for idx, word in enumerate(passage):
                    n.answer += word
                    if idx != len(passage)-1:
                        n.answer += " "
            if n.token == 'spanq':
                start_pos = round(n.child[0].answer.value)
                end_pos =   round(n.child[1].answer.value)
                question = self.get_clear_question()
                question = question.strip().split(' ')
                question = [word.strip() for word in question if word!='']
                question = question[start_pos: end_pos+1]
                n.answer = ""
                for idx, word in enumerate(question):
                    n.answer += word
                    if idx != len(question)-1:
                        n.answer += " "

        elif n.type == 'string':
            n.answer = n.token
        elif n.type == 'number':
            try:
                n.answer = int(n.token)
                n.answer = number_type.number(n.answer)
            except:
                n.answer = self.number2token[n.token]
                if self.diff1flag and type(n.answer) != number_type.percentage:
                    if type(n.answer) == number_type.number:
                        value_temp = n.answer.value 
                        n.answer = number_type.percentage(value_temp,original_text = n.answer.original_text,version = 0)

        elif n.type == 'ROOT':
            n.answer = []##
            self.max_depth = n.depth
            for idx, chil in enumerate(n.child):
                #if not isinstance(chil.answer, str)
                
                n.answer.append(str(chil.answer))
            if len(n.answer) >=4 : # a interesting trick
                n.answer = [str(len(n.answer)),]
        #print(n.type,n.token,n.answer)
    
    def compute_real_program(self, n):
        if n.real_program != None:
            return
        for c in n.child:
            self.compute_real_program(c)
        if n.type == 'operator':
            if n.token in ['add', 'diff','max', 'min','count','argmax','argmin','month','year','day','mul','div','avg','percentage','million']:
                n.real_program = n.token+'('
                for idx, chil in enumerate( n.child):
                    n.real_program += chil.real_program
                    if idx != len(n.child)-1:
                        n.real_program += ','
                n.real_program += ')'
            if n.token == 'key-value':
                n.real_program = '('
                for idx, chil in enumerate(n.child):
                    n.real_program += chil.real_program
                    if idx != len(n.child)-1:
                        n.real_program += ','
                n.real_program += ')'
        elif n.type == 'string':
            n.real_program = "\"" + n.token + "\""
            return
            start_pos, end_pos = self.get_pos(n.token, self.get_clear_question())
            if start_pos != -1:
                n.real_program = 'spanq({},{})'.format(start_pos,end_pos)
                return
            start_pos, end_pos = self.get_pos(n.token, self.get_clear_passage())
            if start_pos != -1:
                n.real_program = 'spanp({},{})'.format(start_pos,end_pos)
                return
            
            question = self.get_clear_question()
            passage = self.get_clear_passage()
            raise Exception(f"question:{question}\n passage:{passage}\n token:{n.token}\n \n")
        elif n.type == 'number':
            n.token = str(n.token)
            if self.old2new_dict != None:
                if n.token in self.old2new_dict:
                    n.token = self.old2new_dict[n.token]
                else :
                    passageId = self.reader.questionId2passageId(self.questionId)
                    passage = self.reader.passageId2add_number_token_passage(passageId)
                    old_passage = self.reader.add_number_token(self.reader.passageId2corenlp_passage(passageId))
                    logger.info(f"passage: {passage}\n old_passageId: {old_passage}\n questionId:{self.questionId}  old_program:{self.program}  invalid_token:{n.token}")
                    #logger.info(f"corenlp passage: {self.reader.passageId2corenlp_passage(passageId)}")
            n.real_program = str(n.token)
        elif n.type == 'ROOT':
            n.real_program = ""
            for idx, chil in enumerate(n.child):
                n.real_program += chil.real_program
                if idx != len(n.child)-1:
                    n.real_program +=','
    
    def get_real_program(self):
        try:
            self.get_program_syntax_tree()
            self.compute_real_program(self.rt)
            return (True,self.rt.real_program)
        except Exception as e :
            errormsg = self.program+" can't get real program!"+" "+str(e)
            return (False,errormsg)
    
    def get_answer(self):
        try:    
            self.get_program_syntax_tree()
            self.compute_answer(self.rt)
            answer = fix_answer(self.reader,self.questionId,self.rt.answer)
            return (True,answer)
        except Exception as e :
            errormsg = self.program+" can't run program!"+" "+str(traceback.format_exc())
            return (False,errormsg)
    
    def get_pos(self, string, passage):
        passage = passage.strip().split(' ')
        string = string.strip().split(' ')
        passage = [word.lower().strip() for word in passage if word!=""]
        string = [word.lower().strip() for word in string if word!=""]
        for start_pos in range(len(passage)):
            if start_pos + len(string) - 1 >= len(passage):
                break
            flag_match = True
            for offset in range(len(string)):
                if string[offset]!= passage[start_pos + offset]:
                    flag_match = False
            if flag_match == True:
                return start_pos, start_pos +len(string) -1
        return -1,-1

    def replace_string(self):
        if_ok , real_program = self.get_real_program()
        return if_ok, real_program




def fix_answer(reader, questionId, answers):
    question = reader.questionId2add_number_token_question_dict[questionId]

    if ('how many million' in question.lower()) or ('how many more million' in question.lower()) or ("how many more" in question.lower() and "in million" in question.lower()):
        for idx, answer in enumerate(answers):
            try:
                answer = str(float(answer)/1000000)
                answers[idx] = answer
            except:
                pass
    if ('how many billion' in question.lower()) or ('how many more billion' in question.lower()) or ("how many more" in question.lower() and "in billion" in question.lower()):
        for idx, answer in enumerate(answers):
            try:
                answer = str(float(answer)/1000000000)
                answers[idx] = answer
            except:
                pass

    if ('what month' in question.lower()):
        for idx, answer in enumerate(answers):
            for month in month_table:
                if month.lower() in answer.lower():
                    answers[idx] = month

    return answers