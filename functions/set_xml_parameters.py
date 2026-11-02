import re

def set_xml_parameters(xml_content, P):
    
    pattern = re.compile(r'\${(\w+)}') # Pattern to find placeholders in the form ${x}.
    
    def replacer(match):
        key = match.group(1)  # Extract the field name (x).
        value = P.get(key, match.group(0))  # Get the corresponding value or keep the placeholder.
        
        if isinstance(value, (int, float)): # If the value is numeric, convert it to a string.
            value = str(value)
        
        return value
    
    return pattern.sub(replacer, xml_content) # Substitute the placeholders in the XML content.