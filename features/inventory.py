import re

def scale_ingredient(ingredient, factor):
    """
    Scale an ingredient quantity by a given factor.
    
    Args:
        ingredient (str): Ingredient string (e.g., "2 cups flour")
        factor (float): Scaling factor (e.g., 2.0 for doubling)
    
    Returns:
        str: Scaled ingredient string
    """
    ingredient = ingredient.strip()
    
    number_pattern = r'(\d+\.?\d*|\d+/\d+)\s*'
    match = re.match(number_pattern, ingredient)
    
    if match:
        quantity_str = match.group(1)
        rest_of_ingredient = ingredient[len(match.group(0)):].strip()
        
        if '/' in quantity_str:
            parts = quantity_str.split('/')
            quantity = (float(parts[0]) / float(parts[1])) * factor
        else:
            quantity = float(quantity_str) * factor
        
        if quantity == int(quantity):
            return f"{int(quantity)} {rest_of_ingredient}"
        else:
            return f"{quantity:.2f} {rest_of_ingredient}".rstrip('0').rstrip('.')
    
    return ingredient


def scale_ingredients(ingredients, original_servings, target_servings):
    """
    Scale multiple ingredients based on serving size change.
    
    Args:
        ingredients (list): List of ingredient strings
        original_servings (int): Original number of servings
        target_servings (int): Target number of servings
    
    Returns:
        list: List of scaled ingredient strings
    """
    scaling_factor = target_servings / original_servings if original_servings > 0 else 1
    
    scaled_ingredients = []
    for ingredient in ingredients:
        scaled_ing = scale_ingredient(ingredient, scaling_factor)
        scaled_ingredients.append(scaled_ing)
    
    return scaled_ingredients
