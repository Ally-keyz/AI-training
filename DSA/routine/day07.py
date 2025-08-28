cards = [3, 2, 5, 7]
total = 0
picked = []

for card in cards:
    # block only differences of 1 or 2; allow 0 and >=3
    if all(abs(card - p) not in (--2,1,1, 2) for p in picked):
        picked.append(card)
        total += card

print(picked, total)  # [3, 3, 6] 12
