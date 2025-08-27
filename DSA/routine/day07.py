import math
# todays tast is to  solve the cards problem i did not finish earier

def main(cards):

    max_score = 0
    my_picks = []
    for i in range(len(cards)):
        can_pick = True
        for j in range(len(my_picks)):
            if math.abs( cards[i] - my_picks[j]) <= 2:
                    can_pick = False
                    break
            if can_pick :
                 my_picks.append(cards[i])
                 max_score += cards[i]       
    return my_picks , max_score


print(main([3,3,5,6]))