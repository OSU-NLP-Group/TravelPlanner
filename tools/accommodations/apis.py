import pandas as pd
from pandas import DataFrame
from typing import Optional
from annotation.src.utils import extract_before_parenthesis


class Accommodations:
    def __init__(self, path="/home/user/app/database/accommodations/clean_accommodations_2022.csv"):
        self.path = path
        self.data = pd.read_csv(self.path).dropna()[['NAME','price','room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number', 'city']]
        print("Accommodations loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == city]
        # results = results[results["date"] == date]
        # if order == "ascPrice":
        #     results = results.sort_values(by=["price"], ascending=True)
        # elif order == "descPrice":
        #     results = results.sort_values(by=["price"], ascending=False)
        # elif order == "ascRate":
        #     results = results.sort_values(by=["review rate number"], ascending=True)
        # elif order == "descRate":
        #     results = results.sort_values(by=["review rate number"], ascending=False)
        # elif order == "ascMinumNights":
        #     results = results.sort_values(by=["minimum nights"], ascending=True)
        # elif order == "descMinumNights":
        #     results = results.sort_values(by=["minimum nights"], ascending=False)
        # elif order == "ascMaximumOccupancy":
        #     results = results.sort_values(by=["maximum occupancy"], ascending=True)
        # elif order == "descMaximumOccupancy":
        #     results = results.sort_values(by=["maximum occupancy"], ascending=False)
        
        # if room_type == "all":
        #     return results
        # elif room_type == "Entire home/apt":
        #     return results[results["room type"]=="Entire home/apt"]
        # elif room_type == "Hotel room":
        #     return results[results["room type"]=="Hotel room"]
        # elif room_type == "Private room":
        #     return results[results["room type"]=="Private room"]
        # elif room_type == "Shared room":
        #     return results[results["room type"]=="Shared room"]
        # else:
        #     return None
        if len(results) == 0:
            return "There is no attraction in this city."
        
        return results
    
    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for accommodations by city."""
        results = self.data[self.data["city"] == extract_before_parenthesis(city)]
        # results = results[results["date"] == date]
        # if order == "ascPrice":
        #     results = results.sort_values(by=["price"], ascending=True)
        # elif order == "descPrice":
        #     results = results.sort_values(by=["price"], ascending=False)
        # elif order == "ascRate":
        #     results = results.sort_values(by=["review rate number"], ascending=True)
        # elif order == "descRate":
        #     results = results.sort_values(by=["review rate number"], ascending=False)
        # elif order == "ascMinumNights":
        #     results = results.sort_values(by=["minimum nights"], ascending=True)
        # elif order == "descMinumNights":
        #     results = results.sort_values(by=["minimum nights"], ascending=False)
        # elif order == "ascMaximumOccupancy":
        #     results = results.sort_values(by=["maximum occupancy"], ascending=True)
        # elif order == "descMaximumOccupancy":
        #     results = results.sort_values(by=["maximum occupancy"], ascending=False)
        
        # if room_type == "all":
        #     return results
        # elif room_type == "Entire home/apt":
        #     return results[results["room type"]=="Entire home/apt"]
        # elif room_type == "Hotel room":
        #     return results[results["room type"]=="Hotel room"]
        # elif room_type == "Private room":
        #     return results[results["room type"]=="Private room"]
        # elif room_type == "Shared room":
        #     return results[results["room type"]=="Shared room"]
        # else:
        #     return None
        return results