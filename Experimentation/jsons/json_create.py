import json

# Directions data as a dictionary
directions = {
    "123 Tech Lane, Innovation City, CA 94016": [
        "Start heading towards the Statue of Liberty.",
        "Turn right at the first McDonald's.",
        "Continue straight past the Central Park entrance.",
        "Turn left onto Broadway Avenue.",
        "Enter the bank's parking garage on your left."
    ],
    "789 Managerial Ave, Innovation City, CA 94016": [
        "Begin by driving towards Central Park.",
        "Turn left at the first CVS Pharmacy.",
        "Drive past the Rockefeller Center.",
        "Turn right onto 5th Avenue.",
        "Enter the bank's parking lot on your right."
    ],
    "456 Creative Blvd, Innovation City, CA 94016": [
        "Head towards Times Square.",
        "Turn right at the first Dunkin' Donuts.",
        "Continue straight past Madison Square Garden.",
        "Turn left onto Lexington Avenue.",
        "Enter the bank's parking structure on your right."
    ],
    "321 Developer Dr, Innovation City, CA 94016": [
        "Head towards the Golden Gate Bridge.",
        "Turn left at the first Starbucks.",
        "Continue straight past the City Hall.",
        "Turn right onto Market Street.",
        "Enter the bank's parking garage on your left."
    ],
    "654 Entrepreneur St, Innovation City, CA 94016": [
        "Begin by driving towards the Empire State Building.",
        "Turn right at the first Walgreens.",
        "Drive past the Lincoln Center.",
        "Turn left onto 7th Avenue.",
        "Enter the bank's parking lot on your right."
    ],
    "987 Visionary Way, Innovation City, CA 94016": [
        "Head towards the Hollywood Sign.",
        "Turn left at the first Subway.",
        "Continue straight past the Griffith Observatory.",
        "Turn right onto Sunset Boulevard.",
        "Enter the bank's parking structure on your left."
    ],
    "159 Innovator Rd, Innovation City, CA 94016": [
        "Start heading towards the Eiffel Tower replica.",
        "Turn right at the first Panera Bread.",
        "Drive past the Museum of Modern Art.",
        "Turn left onto Park Avenue.",
        "Enter the bank's parking garage on your right."
    ],
    "753 Pioneer Pl, Innovation City, CA 94016": [
        "Begin by driving towards the Washington Monument.",
        "Turn left at the first Taco Bell.",
        "Continue straight past the Smithsonian Museum.",
        "Turn right onto Pennsylvania Avenue.",
        "Enter the bank's parking lot on your left."
    ],
    "246 Vision Blvd, Innovation City, CA 94016": [
        "Head towards the Space Needle.",
        "Turn right at the first Chipotle.",
        "Drive past the Pike Place Market.",
        "Turn left onto 1st Avenue.",
        "Enter the bank's parking structure on your right."
    ],
    "369 Inventor St, Innovation City, CA 94016": [
        "Start heading towards the Gateway Arch.",
        "Turn left at the first KFC.",
        "Continue straight past the City Park.",
        "Turn right onto Main Street.",
        "Enter the bank's parking garage on your left."
    ]
}

# Output file path
output_file = "directions.json"

# Write data to JSON file
with open(output_file, 'w') as f:
    json.dump(directions, f, indent=2)

print(f"JSON data written to {output_file}")
