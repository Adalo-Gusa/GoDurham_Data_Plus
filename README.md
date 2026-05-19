
Desktop Inventory Protocol

To record and update our bus stop inventory for comparison to the city's field, we are going to be using data from images obtained from Google Maps street view. To obtain these images, we will implement a Google Maps API scraper to get street view images of every bus stop in Durham from 2025 onward. To analyze the data we will train and implement a machine learning model either through Google Cloud Vision. In the provided spreadsheet, we were given the latitude and longitude of each bus stop in Durham. We will restate this data in our inventory. The attributes we will be collecting include the presence of a (1) shelter, (2) bus sign, (3) GoLive sign, (4) bench, (5) sidewalk, (6) light, (7) trashcan, and (8) ADA pad recorded with a “Yes” or “No” contingent on their presence. We will specify if a trashcan has a lid. The (9) surface type of the stop will also be recorded, with its corresponding value being a string depending on the surface. For example, a stop surrounded by grass would have a surface type value of “grassy.” Further, aiming to avoid violation of Americans with Disabilities Act, we’ll come up with a metric to score the condition based off off other binary indicator, 

These attributes provide an easily comparable overview of the amenities and quality of the bus stops. This would ideally give the city information to identify the bus stops most in need of maintenance and improvements. Furthermore, we will only collect data from street view images taken from 2025 and onward. For obscured or unclear views of the bus stop, the specific attribute will be attributed a NA value with an asterisk which states that the attribute is ambiguous

Attribute Table
Attribute
Description
Values
Shelter
If a shelter is present at the stop
“Yes”, “No”, “N/A”
Bus Sign
If a bus sign is present at the stop


“Yes”, “No”, “N/A”
GoLive
If GoLive info is provided at the stop
“Yes”, “No”, “N/A”
Bench
If bench is present at the stop


“Yes”, “No”, “N/A”
Sidewalk 
If a sidewalk is present throughout the bus stop and surrounding area
“Yes”, “No”, “N/A”
Light 
Entry captures the light condition
“Yes”, “No”, “N/A”
Trashcan
If a Trashcan is present and whether it has a lid
“Yes with lid”, “Yes”, “No”, “N/A”
Stop surface
Entry will be the surface type
“Grass”, “Paved  Pads”
ADA compliance
Is the bus stop compliant to ADA regulations
“Yes”, “No”, “N/A”


Notes:
We need Address from Jenny
What are ADA issues?
