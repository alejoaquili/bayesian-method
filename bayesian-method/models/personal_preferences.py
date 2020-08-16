def generate_string(name, value):
    if value:
        return "Likes " + name + "\n"
    else:
        return "Does not likes " + name + "\n"


class PersonPreference:
    def __init__(self, scones, beer, whiskey, oats, football, nationality):
        self.scones = scones
        self.beer = beer
        self.whiskey = whiskey
        self.oats = oats
        self.football = football
        self.nationality = nationality

    def __str__(self):
        representation = ""
        representation += generate_string("scones", self.scones)
        representation += generate_string("cerveza", self.beer)
        representation += generate_string("wiskey", self.whiskey)
        representation += generate_string("avena", self.oats)
        representation += generate_string("futbol", self.football)
        representation += "Nacionalidad: " + self.nationality + "\n\n"
        return representation
