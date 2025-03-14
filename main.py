import numpy as np

def levenshtein_distance(word1: str, word2: str) -> int:
    len_w1, len_w2 = len(word1), len(word2)
    if len_w1 < len_w2:
        return levenshtein_distance(word2, word1)
    if len_w2 == 0:
        return len_w1
    
    prev_row = np.arange(len_w2 + 1)
    for i, char1 in enumerate(word1, 1):
        curr_row = np.zeros(len_w2 + 1)
        curr_row[0] = i
        for j, char2 in enumerate(word2, 1):
            curr_row[j] = min(
                curr_row[j - 1] + 1,
                prev_row[j] + 1,
                prev_row[j - 1] + (char1 != char2)
            )
        prev_row = curr_row
    
    return int(prev_row[-1])

def damerau_levenshtein_distance(word1: str, word2: str) -> int:
    len_w1, len_w2 = len(word1), len(word2)
    if len_w1 < len_w2:
        return damerau_levenshtein_distance(word2, word1)
    if len_w2 == 0:
        return len_w1
    
    dp = np.zeros((len_w1 + 1, len_w2 + 1), dtype=int)
    dp[0] = np.arange(len_w2 + 1)
    dp[:, 0] = np.arange(len_w1 + 1)
    
    for i in range(1, len_w1 + 1):
        for j in range(1, len_w2 + 1):
            cost = word1[i - 1] != word2[j - 1]
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
            if i > 1 and j > 1 and word1[i - 1] == word2[j - 2] and word1[i - 2] == word2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)
    
    return dp[len_w1][len_w2]

def jaro_winkler_distance(s1: str, s2: str, p: float = 0.1) -> float:
    len_s1, len_s2 = len(s1), len(s2)
    if len_s1 == 0 or len_s2 == 0:
        return 0.0
    
    match_distance = (max(len_s1, len_s2) // 2) - 1
    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2
    
    matches = 0
    transpositions = 0
    
    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
    
    if matches == 0:
        return 0.0
    
    k = 0
    for i in range(len_s1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
    
    transpositions /= 2
    jaro = ((matches / len_s1) + (matches / len_s2) + ((matches - transpositions) / matches)) / 3
    prefix = 0
    for i in range(min(len_s1, len_s2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    
    return jaro + (prefix * p * (1 - jaro))

if __name__ == "__main__":
    word1 = input("Enter the first word: ").strip()
    word2 = input("Enter the second word: ").strip()
    
    lev_dist = levenshtein_distance(word1, word2)
    dam_lev_dist = damerau_levenshtein_distance(word1, word2)
    jaro_wink_dist = jaro_winkler_distance(word1, word2)
    
    print(f"Levenshtein Distance: {lev_dist}")
    print(f"Damerau-Levenshtein Distance: {dam_lev_dist}")
    print(f"Jaro-Winkler Distance: {jaro_wink_dist:.4f}")