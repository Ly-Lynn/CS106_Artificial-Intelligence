def betterEvaluationFunction2(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    foodList = newFood.asList()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    closestGhostDis = float("inf")
    closestFoodDis = float("inf")
    closestCapDis  = float("inf")

    for ghost in newGhostStates:
        dis = manhattanDistance(newPos, ghost.getPosition())
        if closestGhostDis > dis:
            closestGhost = ghost
            closestGhostDis = dis
            closestGhostPos = ghost.getPosition()

    for food in foodList:
        dis = manhattanDistance(newPos, food)
        if dis < closestFoodDis:
            closestFoodDis = dis
            closestFoodPos = food
        
    if newCapsules:
        for caps in newCapsules:
            dis = manhattanDistance(newPos, caps)
            if dis < closestCapDis:
                closestCapsule = dis
                closestCapPos = caps
    else:
        closestCapsule = 0

    # capsuleghostDis = manhattanDistance(closestGhost, closestCapPos) * closestCapsule
    if closestCapsule:
        closest_capsule = -2 / closestCapsule
    else:
        closest_capsule = 100
    
    if closestGhost.scaredTimer > 0:
        ghost_distance = 1 / closestGhostDis #càng gần scared ghost điểm càng cao
        ghost_distance *= (closestGhost.scaredTimer / 40)  # Điểm cao hơn nếu ghost còn sợ hãi
        # leftFood = 10
        closestFoodDis = 2 / closestFoodDis
    else:
        if closestGhostDis:
            ghost_distance = -2 / closestGhostDis
        else:
            ghost_distance = -500
        # foodghostDis = manhattanDistance(closestGhostPos, closestFoodPos)
        if closestFoodDis:
            foodScores = -1 / closestFoodDis # khoang cach giua food gan nhat và ghost gần nhất càng xa nhau thì càng ưu tiên ăn food
            # leftFood = -1
        else: # ngược lại thì ưu tiên né ghost và move to another food
            foodScores= 0
            # ghost_distance *= 2
            # leftFood = -5
    
    score =  foodScores + ghost_distance -10  * len(foodList) + closest_capsule

    print(newScaredTimes, score, closest_capsule, sep=" ")
    # return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule
    # print(score)
    return score
def betterEvaluationFunction2(currentGameState):
    """
    This function evaluates the current game state for Pacman.
    
    It considers the distances to the closest ghost, closest food, and closest capsule,
    as well as the remaining scared time of the ghosts and the amount of remaining food.
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    foodList = newFood.asList()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()

    # Initialize distances to infinity
    closest_ghost_dis = float("inf")
    closest_food_dis = float("inf")
    closest_cap_dis = float("inf")
    
    # Calculate distances to the closest ghost, food, and capsule
    for ghost in newGhostStates:
        ghost_dis = manhattanDistance(newPos, ghost.getPosition())
        closest_ghost_dis = min(closest_ghost_dis, ghost_dis)
    
    for food in foodList:
        food_dis = manhattanDistance(newPos, food)
        closest_food_dis = min(closest_food_dis, food_dis)
        
    for caps in newCapsules:
        cap_dis = manhattanDistance(newPos, caps)
        closest_cap_dis = min(closest_cap_dis, cap_dis)
    
    # Initialize scores
    ghost_distance_score = 0
    food_distance_score = 0
    remaining_food_score = 0
    capsule_score = 0
    
    # Calculate ghost distance score
    if closest_ghost_dis:
        if newGhostStates[0].scaredTimer > 0:
            ghost_distance_score = 1 / closest_ghost_dis
            ghost_distance_score *= (newGhostStates[0].scaredTimer / 40)
        else:
            ghost_distance_score = -2 / closest_ghost_dis
    else:
        ghost_distance_score = -500
    
    # Calculate food distance score
    if closest_food_dis:
        food_distance_score = -1 / closest_food_dis
    else:
        food_distance_score = 0
    
    # Calculate remaining food score
    remaining_food_score = -5 * len(foodList)
    
    # Calculate capsule score
    if closest_cap_dis:
        capsule_score = -2 / closest_cap_dis
    else:
        capsule_score = 100
    
    # Calculate total score
    total_score = ghost_distance_score + food_distance_score + remaining_food_score + capsule_score
    
    return total_score