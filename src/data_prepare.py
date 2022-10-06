from utils import getNumberFromString
from number_type import getNumber
import re
import logging
logger = logging.getLogger()



def old_number2token(passage, NorQ = 'N'):
    
    number2token_ = {}
    useful_ner = ['ORDINAL','DATE', 'NUMBER', 'MONEY','DURATION','PERCENT','TIME']
    num = 0
    for sentence in passage['sentences']:
        pre = []
        for id_entity, entity in enumerate(sentence['entitymentions']):
            if entity['ner'] in useful_ner:
                if id_entity+1 < len(sentence['entitymentions']):
                    next_entity = sentence['entitymentions'][id_entity+1]
                    if next_entity['tokenBegin'] == entity['tokenEnd'] and next_entity['ner'] == entity['ner']:
                        pre.append(entity)
                        continue
                
                def getNumType(Entity):
                    normalizedNER = ''
                    if 'normalizedNER' in Entity:
                        normalizedNER = Entity['normalizedNER']
                    number = getNumber(Entity)
                    return number

                if pre !=[] :
                    pre.append(entity)
                    pre = [entity for entity in pre if entity['text'].lower()!='and']
                    for idx, e in enumerate(pre):
                        number2token_[NorQ+str(num)+'_'+str(idx)] = getNumType(e)
                    pre = []
                else :
                    number2token_[NorQ+str(num)] = getNumType(entity)
                    list_number = getNumberFromString(entity['text'])
                    #print(entity['text'], list_number)
                    if len(list_number) > 1:
                        for idx, i in enumerate(list_number):
                            number2token_[NorQ+str(num)+'_'+str(idx)] = getNumber(i)
                num += 1

    for i in range(11):
        number2token_['X'+str(i)] = getNumber(float(i))
    return number2token_


def new_number2token(passage, NorQ = 'N', debug_mode = False):
    number2token_ = {}
    useful_ner = ['ORDINAL','DATE', 'NUMBER', 'MONEY','DURATION','PERCENT','TIME']
    FLAG_NER = '@'

    new_num = 0
    old_num = 0
    old2new_dict = {}
    idx2token = {}
    invalid_token = {}

    for sentence in passage['sentences']:
        pre = []
        idx2token[sentence['index']] = {}
        invalid_token[sentence['index']] = {}
        for id_entity, entity in enumerate(sentence['entitymentions']):
            entity['text'] = re.sub(u'\u2013','-',entity['text'])
            
            if entity['ner'] in useful_ner:
                if id_entity+1 < len(sentence['entitymentions']):
                    next_entity = sentence['entitymentions'][id_entity+1]
                    if next_entity['tokenBegin'] == entity['tokenEnd'] and next_entity['ner'] == entity['ner']:
                        pre.append(entity)
                        continue
                def getNumType(Entity):
                    number = getNumber(Entity)
                    return number
                if pre !=[] :
                    pre.append(entity)
                    pre = [entity for entity in pre if entity['text'].lower()!='and']
                    for idx, e in enumerate(pre):
                        number2token_[NorQ+str(new_num)] = getNumType(e)
                        old2new_dict[NorQ+str(old_num)+ '_' + str(idx)] = NorQ +str(new_num)
                        idx2token[sentence['index']][e['tokenEnd'] - 1] = NorQ + str(new_num) 
                        new_num += 1
                    pre = []
                else :
                    if len(getNumberFromString(entity['text'])) <= 1 or (not ('-' in entity['text'] or ':' in entity['text'] or '/' in entity['text'])): 
                        number2token_[NorQ+str(new_num)] = getNumType(entity)
                        idx2token[sentence['index']][entity['tokenEnd'] - 1] = NorQ + str(new_num) 
                        old2new_dict[NorQ + str(old_num)] = NorQ + str(new_num)
                        new_num += 1

                    else : 
                        text = entity['text']
                        if True:
                            text_temp = re.sub(' ','',entity['text'])
                            
                            if re.match(u'^[0-9]{4}[^0-9a-zA-Z]{1}[0-9]{2}$',text_temp):
                                #print(text_temp)
                                text_temp = [text_temp[0:4],text_temp[5:7]]
                                num = [0,0,0,0]
                                num[0] = int(text_temp[0])
                                num[1] = int(text_temp[0][2:4])
                                num[2] = int(text_temp[1])
                                if num[1] < num[2] and num[0] <= 2020: 
                                    text = text_temp[0] + '-' + text_temp[0][0:2] + text_temp[1] 
                                    #print(text)
                            elif re.match(u'^[0-9]{4}[^0-9a-zA-Z]{1}[0-9]{1}$',text_temp):
                                #print(text_temp)
                                text_temp = [text_temp[0:4],text_temp[5:6]]
                                num = [0,0,0,0]
                                num[0] = int(text_temp[0])
                                num[1] = int(text_temp[0][3:4])
                                num[2] = int(text_temp[1])
                                if num[1] < num[2] and num[0] <= 2020: 
                                    text = text_temp[0] + '-' + text_temp[0][0:3] + text_temp[1]
                                    #print(text)
                        iter = re.finditer("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", text)
                        num_indexs = [(i.group(), i.span()) for i in iter]
                        len_ins = 0
                        
                        
                        for i in range(entity['tokenBegin'],entity['tokenEnd']):
                            invalid_token[sentence['index']][i] = ""
                        for idx,( num,num_idx) in enumerate(num_indexs):
                            try:
                                num = abs(float(re.sub(',','',num)))
                            except:
                                num = 0.0
                            number2token_[NorQ+str(new_num)] = getNumber(num)
                            old2new_dict[NorQ+str(old_num)+ '_' + str(idx)] = NorQ +str(new_num)
                            inserted_string = FLAG_NER + NorQ + str(new_num) + " "
                            new_num += 1

                            text = text[: len_ins + num_idx[1]] + inserted_string + text[ len_ins + num_idx[1]:]
                            len_ins += len(inserted_string)
                        text = ' ' + text
                        invalid_token[sentence['index']][entity['tokenBegin']] = text 
                        old2new_dict[NorQ+ str(old_num)] = NorQ + str(new_num-1)
                old_num += 1

    for i in range(11):
        number2token_['X'+str(i)] = getNumber(i)
        old2new_dict['X'+str(i)] = 'X' + str(i)
    old2new_dict['1'] = '1'
    return number2token_

