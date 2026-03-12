def spiral_order(matrix: list[list[int]]) -> list[int]:
    if not matrix or not matrix[0]:
        return []
    top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    result = []
    while top <= bottom and left <= right:
        result.extend(matrix[top][left:right + 1])
        for r in range(top + 1, bottom + 1):
            result.append(matrix[r][right])
        if top < bottom and left < right:
            result.extend(matrix[bottom][right - 1:left - 1 if left else None:-1])
        if left < right:
            for r in range(bottom - 1, top, -1):
                result.append(matrix[r][left])
        top, bottom, left, right = top + 1, bottom - 1, left + 1, right - 1
    return result
