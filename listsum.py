l1 = [7,4,2,3]
l2 = [9,5,6]

class listnode():
    def __init__(self,val,next = None):
        self.val = val
        self.next = next


l11 = listnode(3)
l12 = listnode(2,l11)
l13 = listnode(4,l12)
l14 = listnode(7,l13)

l21 = listnode(6)
l22 = listnode(5,l21)
l23 = listnode(9,l22)

def reverse(head):
    pred = None
    temp = head
    while temp != None:
        next=  temp.next
        temp.next = pred
        pred = temp
        temp = next
    return pred

new_l1 = reverse(l23)
while(new_l1 != None):
    print(new_l1.val)
    new_l1 = new_l1.next

def get_sum(l1,l2):
    l1v = reverse(l1)
    l2v = reverse(l2)
    sum = 0
    carry = 0
    pred = None
    while((l1v != None) and (l2v != None)):
        temp_val = l1v.val+l2v.val+carry
        if temp_val < 10:
            temp = listnode(temp_val,pred)
            carry = 0
        else:
            temp = listnode(temp_val -10,pred)
            carry = temp_val-10
        pred = temp

        l1v = l1v.next
        l2v = l2v.next
    while(l1v!= None):
        #print(l1v.val)
        temp_val = l1v.val + carry
        if temp_val < 10:
            temp = listnode(temp_val,pred)
            carry = 0
        else:
            temp = listnode(temp_val - 10, pred)
            carry = temp_val - 10
        pred = temp
        l1v = l1v.next
    while (l2v != None):
        temp_val = l2v.val + carry
        if temp_val < 10:
            temp = listnode(temp_val, pred)
            carry = 0
        else:
            temp = listnode(temp_val - 10, pred)
            carry = temp_val - 10
        pred = temp
        l2v = l2v.next
    return pred

sum_list= get_sum(l14,l23)
while(sum_list != None):
    #print(sum_list.val)
    sum_list = sum_list.next


a = [1,2,3,4,5,6]