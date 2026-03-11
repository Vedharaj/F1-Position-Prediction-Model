import difflib
from io import StringIO
import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide"
)

# -----------------------
# Load model
# -----------------------
model = joblib.load("f1_position_model.pkl")

# -----------------------
# Dictionaries (name → id)
# -----------------------

drivers= {'Lewis Hamilton': 1, 'Nick Heidfeld': 2, 'Nico Rosberg': 3, 'Fernando Alonso': 4, 'Heikki Kovalainen': 5, 'Kazuki Nakajima': 6, 'Sébastien Bourdais': 7, 'Kimi Räikkönen': 8, 'Robert Kubica': 9, 'Timo Glock': 10, 'Takuma Sato': 11, 'Nelson Piquet Jr.': 12, 'Felipe Massa': 13, 'David Coulthard': 14, 'Jarno Trulli': 15, 'Adrian Sutil': 16, 'Mark Webber': 17, 'Jenson Button': 18, 'Anthony Davidson': 19, 'Sebastian Vettel': 20, 'Giancarlo Fisichella': 21, 'Rubens Barrichello': 22, 'Ralf Schumacher': 23, 'Vitantonio Liuzzi': 24, 'Alexander Wurz': 25, 'Scott Speed': 26, 'Christijan Albers': 27, 'Markus Winkelhock': 28, 'Sakon Yamamoto': 29, 'Michael Schumacher': 30, 'Juan Pablo Montoya': 31, 'Christian Klien': 32, 'Tiago Monteiro': 33, 'Yuji Ide': 34, 'Jacques Villeneuve': 35, 'Franck Montagny': 36, 'Pedro de la Rosa': 37, 'Robert Doornbos': 38, 'Narain Karthikeyan': 39, 'Patrick Friesacher': 40, 'Ricardo Zonta': 41, 'Antônio Pizzonia': 42, 'Cristiano da Matta': 43, 'Olivier Panis': 44, 'Giorgio Pantano': 45, 'Gianmaria Bruni': 46, 'Zsolt Baumgartner': 47, 'Marc Gené': 48, 'Heinz-Harald Frentzen': 49, 'Jos Verstappen': 50, 'Justin Wilson': 51, 'Ralph Firman': 52, 'Nicolas Kiesa': 53, 'Luciano Burti': 54, 'Jean Alesi': 55, 'Eddie Irvine': 56, 'Mika Häkkinen': 57, 'Tarso Marques': 58, 'Enrique Bernoldi': 59, 'Gastón Mazzacane': 60, 'Tomáš Enge': 61, 'Alex Yoong': 62, 'Mika Salo': 63, 'Pedro Diniz': 64, 'Johnny Herbert': 65, 'Allan McNish': 66, 'Sébastien Buemi': 67, 'Toranosuke Takagi': 68, 'Luca Badoer': 69, 'Alessandro Zanardi': 70, 'Damon Hill': 71, 'Stéphane Sarrazin': 72, 'Ricardo Rosset': 73, 'Esteban Tuero': 74, 'Shinji Nakano': 75, 'Jan Magnussen': 76, 'Gerhard Berger': 77, 'Nicola Larini': 78, 'Ukyo Katayama': 79, 'Vincenzo Sospiri': 80, 'Gianni Morbidelli': 81, 'Norberto Fontana': 82, 'Pedro Lamy': 83, 'Martin Brundle': 84, 'Andrea Montermini': 85, 'Giovanni Lavaggi': 86, 'Mark Blundell': 87, 'Aguri Suzuki': 88, 'Taki Inoue': 89, 'Roberto Moreno': 90, 'Karl Wendlinger': 91, 'Bertrand Gachot': 92, 'Domenico Schiattarella': 93, 'Pierluigi Martini': 94, 'Nigel Mansell': 95, 'Jean-Christophe Boullion': 96, 'Massimiliano Papis': 97, 'Jean-Denis Délétraz': 98, 'Gabriele Tarquini': 99, 'Érik Comas': 100, 'David Brabham': 101, 'Ayrton Senna': 102, 'Éric Bernard': 103, 'Christian Fittipaldi': 104, 'Michele Alboreto': 105, 'Olivier Beretta': 106, 'Roland Ratzenberger': 107, 'Paul Belmondo': 108, 'Jyrki Järvilehto': 109, 'Andrea de Cesaris': 110, 'Jean-Marc Gounon': 111, 'Philippe Alliot': 112, 'Philippe Adams': 113, 'Yannick Dalmas': 114, 'Hideki Noda': 115, 'Franck Lagorce': 116, 'Alain Prost': 117, 'Derek Warwick': 118, 'Riccardo Patrese': 119, 'Fabrizio Barbazza': 120, 'Michael Andretti': 121, 'Ivan Capelli': 122, 'Thierry Boutsen': 123, 'Marco Apicella': 124, 'Emanuele Naspetti': 125, 'Toshio Suzuki': 126, 'Maurício Gugelmin': 127, 'Eric van de Poele': 128, 'Olivier Grouillard': 129, 'Andrea Chiesa': 130, 'Stefano Modena': 131, 'Giovanna Amati': 132, 'Alex Caffi': 133, 'Enrico Bertaggia': 134, 'Perry McCarthy': 135, 'Jan Lammers': 136, 'Nelson Piquet': 137, 'Satoru Nakajima': 138, 'Emanuele Pirro': 139, 'Stefan Johansson': 140, 'Julian Bailey': 141, 'Pedro Chaves': 142, 'Michael Bartels': 143, 'Naoki Hattori': 144, 'Alessandro Nannini': 145, 'Bernd Schneider': 146, 'Paolo Barilla': 147, 'Gregor Foitek': 148, 'Claudio Langes': 149, 'Gary Brabham': 150, 'Martin Donnelly': 151, 'Bruno Giacomelli': 152, 'Jaime Alguersuari': 153, 'Romain Grosjean': 154, 'Kamui Kobayashi': 155, 'Jonathan Palmer': 156, 'Christian Danner': 157, 'Eddie Cheever': 158, 'Luis Pérez-Sala': 159, 'Piercarlo Ghinzani': 160, 'Volker Weidler': 161, 'Pierre-Henri Raphanel': 162, 'René Arnoux': 163, 'Joachim Winkelhock': 164, 'Oscar Larrauri': 165, 'Philippe Streiff': 166, 'Adrián Campos': 167, 'Jean-Louis Schlesser': 168, 'Pascal Fabre': 169, 'Teo Fabi': 170, 'Franco Forini': 171, 'Jacques Laffite': 172, 'Elio de Angelis': 173, 'Johnny Dumfries': 174, 'Patrick Tambay': 175, 'Marc Surer': 176, 'Keke Rosberg': 177, 'Alan Jones': 178, 'Huub Rothengatter': 179, 'Allen Berg': 180, 'Manfred Winkelhock': 181, 'Niki Lauda': 182, 'François Hesnault': 183, 'Mauro Baldi': 184, 'Stefan Bellof': 185, 'Kenny Acheson': 186, 'John Watson': 187, 'Johnny Cecotto': 188, 'Jo Gartner': 189, 'Corrado Fabi': 190, 'Mike Thackwell': 191, 'Chico Serra': 192, 'Danny Sullivan': 193, 'Eliseo Salazar': 194, 'Roberto Guerrero': 195, 'Raul Boesel': 196, 'Jean-Pierre Jarier': 197, 'Jacques Villeneuve Sr.': 198, 'Carlos Reutemann': 199, 'Jochen Mass': 200, 'Slim Borgudd': 201, 'Didier Pironi': 202, 'Gilles Villeneuve': 203, 'Riccardo Paletti': 204, 'Brian Henton': 205, 'Derek Daly': 206, 'Mario Andretti': 207, 'Emilio de Villota': 208, 'Geoff Lees': 209, 'Tommy Byrne': 210, 'Rupert Keegan': 211, 'Hector Rebaque': 212, 'Beppe Gabbiani': 213, 'Kevin Cogan': 214, 'Miguel Ángel Guerra': 215, 'Siegfried Stohr': 216, 'Ricardo Zunino': 217, 'Ricardo Londoño': 218, 'Jean-Pierre Jabouille': 219, 'Giorgio Francia': 220, 'Patrick Depailler': 221, 'Jody Scheckter': 222, 'Clay Regazzoni': 223, 'Emerson Fittipaldi': 224, 'Dave Kennedy': 225, 'Stephen South': 226, 'Tiff Needell': 227, 'Desiré Wilson': 228, 'Harald Ertl': 229, 'Vittorio Brambilla': 230, 'James Hunt': 231, 'Arturo Merzario': 232, 'Hans-Joachim Stuck': 233, 'Gianfranco Brancatelli': 234, 'Jacky Ickx': 235, 'Patrick Gaillard': 236, 'Alex Ribeiro': 237, 'Ronnie Peterson': 238, 'Brett Lunger': 239, 'Danny Ongais': 240, 'Lamberto Leoni': 241, 'Divina Galica': 242, 'Rolf Stommelen': 243, 'Alberto Colombo': 244, 'Tony Trimmer': 245, 'Hans Binder': 246, 'Michael Bleekemolen': 247, 'Carlo Franchi': 248, 'Bobby Rahal': 249, 'Carlos Pace': 250, 'Ian Scheckter': 251, 'Tom Pryce': 252, 'Ingo Hoffmann': 253, 'Renzo Zorzi': 254, 'Gunnar Nilsson': 255, 'Larry Perkins': 256, 'Boy Lunger': 257, 'Patrick Nève': 258, 'David Purley': 259, 'Conny Andersson': 260, 'Bernard de Dryver': 261, 'Jackie Oliver': 262, 'Mikko Kozarowitzky': 263, 'Andy Sutcliffe': 264, 'Guy Edwards': 265, 'Brian McGuire': 266, 'Vern Schuppan': 267, 'Hans Heyer': 268, 'Teddy Pilette': 269, 'Ian Ashley': 270, 'Loris Kessel': 271, 'Kunimitsu Takahashi': 272, 'Kazuyoshi Hoshino': 273, 'Noritake Takahara': 274, 'Lella Lombardi': 275, 'Bob Evans': 276, 'Michel Leclère': 277, 'Chris Amon': 278, 'Emilio Zapico': 279, 'Henri Pescarolo': 280, 'Jac Nelleman': 281, 'Damien Magee': 282, 'Mike Wilds': 283, 'Alessandro Pesenti-Rossi': 284, 'Otto Stuppacher': 285, 'Warwick Brown': 286, 'Masahiro Hasemi': 287, 'Mark Donohue': 288, 'Graham Hill': 289, 'Wilson Fittipaldi': 290, 'Guy Tunmer': 291, 'Eddie Keizan': 292, 'Dave Charlton': 293, 'Tony Brise': 294, 'Roelof Wunderink': 295, 'François Migault': 296, 'Torsten Palm': 297, 'Gijs van Lennep': 298, 'Hiroshi Fushida': 299, 'John Nicholson': 300, 'Dave Morgan': 301, 'Jim Crawford': 302, 'Jo Vonlanthen': 303, 'Denny Hulme': 304, 'Mike Hailwood': 305, 'Jean-Pierre Beltoise': 306, 'Howden Ganley': 307, 'Richard Robarts': 308, 'Peter Revson': 309, 'Paddy Driver': 310, 'Tom Belsø': 311, 'Brian Redman': 312, 'Rikky von Opel': 313, 'Tim Schenken': 314, 'Gérard Larrousse': 315, 'Leo Kinnunen': 316, 'Reine Wisell': 317, 'Bertil Roos': 318, 'José Dolhem': 319, 'Peter Gethin': 320, 'Derek Bell': 321, 'David Hobbs': 322, 'Dieter Quester': 323, 'Helmuth Koinigg': 324, 'Carlo Facetti': 325, 'Eppie Wietzes': 326, 'François Cevert': 327, 'Jackie Stewart': 328, 'Mike Beuttler': 329, 'Nanni Galli': 330, 'Luiz Bueno': 331, 'George Follmer': 332, 'Andrea de Adamich': 333, 'Jackie Pretorius': 334, 'Roger Williamson': 335, 'Graham McRae': 336, 'Helmut Marko': 337, 'David Walker': 338, 'Alex Soler-Roig': 339, 'John Love': 340, 'John Surtees': 341, 'Skip Barber': 342, 'Bill Brack': 343, 'Sam Posey': 344, 'Pedro Rodríguez': 345, 'Jo Siffert': 346, 'Jo Bonnier': 347, 'François Mazet': 348, 'Max Jean': 349, 'Vic Elford': 350, 'Silvio Moser': 351, 'George Eaton': 352, 'Pete Lovely': 353, 'Chris Craft': 354, 'John Cannon': 355, 'Jack Brabham': 356, 'John Miles': 357, 'Jochen Rindt': 358, 'Johnny Servoz-Gavin': 359, 'Bruce McLaren': 360, 'Piers Courage': 361, 'Peter de Klerk': 362, 'Ignazio Giunti': 363, 'Dan Gurney': 364, 'Hubert Hahne': 365, 'Gus Hutchison': 366, 'Peter Westbury': 367, 'Sam Tingle': 368, 'Basil van Rooyen': 369, 'Richard Attwood': 370, 'Al Pease': 371, 'John Cordts': 372, 'Jim Clark': 373, 'Mike Spence': 374, 'Ludovico Scarfiotti': 375, 'Lucien Bianchi': 376, 'Jo Schlesser': 377, 'Robin Widdows': 378, 'Kurt Ahrens': 379, 'Frank Gardner': 380, 'Bobby Unser': 381, 'Moisés Solana': 382, 'Bob Anderson': 383, 'Luki Botha': 384, 'Lorenzo Bandini': 385, 'Richie Ginther': 386, 'Mike Parkes': 387, 'Chris Irwin': 388, 'Guy Ligier': 389, 'Alan Rees': 390, 'Brian Hart': 391, 'Mike Fisher': 392, 'Tom Jones': 393, 'Giancarlo Baghetti': 394, 'Jonathan Williams': 395, 'Bob Bondurant': 396, 'Peter Arundell': 397, 'Vic Wilson': 398, 'John Taylor': 399, 'Chris Lawrence': 400, 'Trevor Taylor': 401, 'Giacomo Russo': 402, 'Phil Hill': 403, 'Innes Ireland': 404, 'Ronnie Bucknum': 405, 'Paul Hawkins': 406, 'David Prophet': 407, 'Tony Maggs': 408, 'Trevor Blokdyk': 409, 'Neville Lederle': 410, 'Doug Serrurier': 411, 'Brausch Niemann': 412, 'Ernie Pieterse': 413, 'Clive Puzey': 414, 'Ray Reed': 415, 'David Clapham': 416, 'Alex Blignaut': 417, 'Masten Gregory': 418, 'John Rhodes': 419, 'Ian Raby': 420, 'Alan Rollinson': 421, 'Brian Gubby': 422, 'Gerhard Mitter': 423, 'Roberto Bussinello': 424, 'Nino Vaccarella': 425, 'Giorgio Bassi': 426, 'Maurice Trintignant': 427, 'Bernard Collomb': 428, 'André Pilette': 429, 'Carel Godin de Beaufort': 430, 'Edgar Barth': 431, 'Mário de Araújo Cabral': 432, 'Walt Hansgen': 433, 'Hap Sharp': 434, 'Willy Mairesse': 435, 'John Campbell-Jones': 436, 'Ian Burgess': 437, 'Tony Settember': 438, 'Nasif Estéfano': 439, 'Jim Hall': 440, 'Tim Parnell': 441, 'Kurt Kuhnke': 442, 'Ernesto Brambilla': 443, 'Roberto Lippi': 444, 'Günther Seiffert': 445, 'Carlo Abate': 446, 'Gaetano Starrabba': 447, 'Peter Broeker': 448, 'Rodger Ward': 449, 'Ernie de Vos': 450, 'Frank Dochnal': 451, 'Thomas Monarch': 452, 'Pierre Gasly': 842, 'Jackie Lewis': 453, 'Ricardo Rodríguez': 454, 'Wolfgang Seidel': 455, 'Roy Salvadori': 456, 'Ben Pon': 457, 'Rob Slotemaker': 458, 'Tony Marsh': 459, 'Gerry Ashmore': 460, 'Heinz Schiller': 461, 'Colin Davis': 462, 'Jay Chamberlain': 463, 'Tony Shelly': 464, 'Keith Greene': 465, 'Heini Walter': 466, 'Ernesto Prinoth': 467, 'Roger Penske': 468, 'Rob Schroeder': 469, 'Timmy Mayer': 470, 'Bruce Johnstone': 471, 'Mike Harris': 472, 'Gary Hocking': 473, 'Syd van der Vyver': 474, 'Stirling Moss': 475, 'Wolfgang von Trips': 476, 'Cliff Allison': 477, 'Hans Herrmann': 478, 'Tony Brooks': 479, 'Michael May': 480, 'Henry Taylor': 481, 'Olivier Gendebien': 482, 'Giorgio Scarlatti': 483, 'Brian Naylor': 484, 'Juan Manuel Bordeu': 485, 'Jack Fairman': 486, 'Massimo Natili': 487, 'Peter Monteverdi': 488, 'Renato Pirocchi': 489, 'Geoff Duke': 490, 'Alfonso Thiele': 491, 'Menato Boffa': 492, 'Peter Ryan': 493, 'Lloyd Ruby': 494, 'Ken Miles': 495, 'Carlos Menditeguy': 496, 'Alberto Rodriguez Larreta': 497, 'José Froilán González': 498, 'Roberto Bonomi': 499, 'Gino Munaron': 500, 'Harry Schell': 501, 'Alan Stacey': 502, 'Ettore Chimeri': 503, 'Antonio Creus': 504, 'Chris Bristow': 505, 'Bruce Halford': 506, 'Chuck Daigh': 507, 'Lance Reventlow': 508, 'Jim Rathmann': 509, 'Paul Goldsmith': 510, 'Don Branson': 511, 'Johnny Thomson': 512, 'Eddie Johnson': 513, 'Bob Veith': 514, 'Bud Tingelstad': 515, 'Bob Christie': 516, 'Red Amick': 517, 'Duane Carter': 518, 'Bill Homeier': 519, 'Gene Hartley': 520, 'Chuck Stevenson': 521, 'Bobby Grim': 522, 'Shorty Templeman': 523, 'Jim Hurtubise': 524, 'Jimmy Bryan': 525, 'Troy Ruttman': 526, 'Eddie Sachs': 527, 'Don Freeland': 528, 'Tony Bettenhausen': 529, 'Wayne Weiler': 530, 'Anthony Foyt': 531, 'Eddie Russo': 532, 'Johnny Boyd': 533, 'Gene Force': 534, 'Jim McWithey': 535, 'Len Sutton': 536, 'Dick Rathmann': 537, 'Al Herman': 538, 'Dempsey Wilson': 539, 'Mike Taylor': 540, 'Ron Flockhart': 541, 'David Piper': 542, 'Giulio Cabianca': 543, 'Piero Drogo': 544, 'Fred Gamble': 545, 'Arthur Owen': 546, 'Horace Gould': 547, 'Bob Drake': 548, 'Ivor Bueb': 549, 'Alain de Changy': 550, 'Maria de Filippis': 551, 'Jean Lucienbonnet': 552, 'André Testut': 553, 'Jean Behra': 554, 'Paul Russo': 555, 'Jimmy Daywalt': 556, 'Chuck Arnold': 557, 'Al Keller': 558, 'Pat Flaherty': 559, 'Bill Cheesbourg': 560, 'Ray Crawford': 561, 'Jack Turner': 562, 'Chuck Weyant': 563, 'Jud Larson': 564, 'Mike Magill': 565, 'Carroll Shelby': 566, "Fritz d'Orey": 567, 'Azdrubal Fontes': 568, 'Peter Ashdown': 569, 'Bill Moss': 570, 'Dennis Taylor': 571, 'Harry Blanchard': 572, 'Alessandro de Tomaso': 573, 'George Constantine': 574, 'Bob Said': 575, 'Phil Cade': 576, 'Luigi Musso': 577, 'Mike Hawthorn': 578, 'Juan Fangio': 579, 'Paco Godia': 580, 'Peter Collins': 581, 'Ken Kavanagh': 582, 'Gerino Gerini': 583, 'Bruce Kessler': 584, 'Paul Emery': 585, 'Luigi Piotti': 586, 'Bernie Ecclestone': 587, 'Luigi Taramazzo': 588, 'Louis Chiron': 589, 'Stuart Lewis-Evans': 590, 'George Amick': 591, 'Jimmy Reece': 592, 'Johnnie Parsons': 593, 'Johnnie Tolan': 594, 'Billy Garrett': 595, 'Ed Elisian': 596, "Pat O'Connor": 597, 'Jerry Unser': 598, 'Art Bisch': 599, 'Christian Goethals': 600, 'Dick Gibson': 601, 'Robert La Caze': 602, 'André Guelfi': 603, 'François Picard': 604, 'Tom Bridger': 605, 'Alfonso de Portago': 606, 'Cesare Perdisa': 607, 'Eugenio Castellotti': 608, 'André Simon': 609, 'Les Leston': 610, 'Sam Hanks': 611, 'Andy Linden': 612, 'Marshall Teague': 613, 'Don Edmunds': 614, 'Fred Agabashian': 615, 'Elmer George': 616, 'Mike MacDowel': 617, 'Herbert MacKay-Fraser': 618, 'Bob Gerard': 619, 'Umberto Maglioli': 620, 'Paul England': 621, 'Chico Landi': 622, 'Alberto Uria': 623, 'Hernando da Silva Ramos': 624, 'Élie Bayol': 625, 'Robert Manzon': 626, 'Louis Rosier': 627, 'Bob Sweikert': 628, 'Cliff Griffith': 629, 'Duke Dinsmore': 630, 'Keith Andrews': 631, 'Paul Frère': 632, 'Luigi Villoresi': 633, 'Piero Scotti': 634, 'Colin Chapman': 635, 'Desmond Titterington': 636, 'Archie Scott Brown': 637, 'Ottorino Volonterio': 638, 'André Milhoux': 639, 'Toulo de Graffenried': 640, 'Piero Taruffi': 641, 'Nino Farina': 642, 'Roberto Mieres': 643, 'Sergio Mantovani': 644, 'Clemar Bucci': 645, 'Jesús Iglesias': 646, 'Alberto Ascari': 647, 'Karl Kling': 648, 'Pablo Birger': 649, 'Jacques Pollet': 650, 'Lance Macklin': 651, 'Ted Whiteaway': 652, 'Jimmy Davies': 653, 'Walt Faulkner': 654, 'Cal Niday': 655, 'Art Cross': 656, 'Bill Vukovich': 657, 'Jack McGrath': 658, 'Jerry Hoyt': 659, 'Johnny Claes': 660, 'Peter Walker': 661, 'Mike Sparken': 662, 'Ken Wharton': 663, 'Kenneth McAlpine': 664, 'Leslie Marr': 665, 'Tony Rolt': 666, 'John Fitch': 667, 'Jean Lucas': 668, 'Prince Bira': 669, 'Onofre Marimón': 670, 'Roger Loyer': 671, 'Jorge Daponte': 672, 'Mike Nazaruk': 673, 'Larry Crockett': 674, 'Manny Ayulo': 675, 'Frank Armi': 676, 'Travis Webb': 677, 'Len Duncan': 678, 'Ernie McCoy': 679, 'Jacques Swaters': 680, 'Georges Berger': 681, 'Don Beauman': 682, 'Leslie Thorne': 683, 'Bill Whitehouse': 684, 'John Riseley-Prichard': 685, 'Reg Parnell': 686, 'Peter Whitehead': 687, 'Eric Brandon': 688, 'Alan Brown': 689, 'Rodney Nuckey': 690, 'Hermann Lang': 691, 'Theo Helfrich': 692, 'Fred Wacker': 693, 'Giovanni de Riu': 694, 'Oscar Gálvez': 695, 'John Barber': 696, 'Felice Bonetto': 697, 'Adolfo Cruz': 698, 'Duke Nalon': 699, 'Carl Scarborough': 700, 'Bill Holland': 701, 'Bob Scott': 702, 'Arthur Legat': 703, 'Yves Cabantous': 704, 'Tony Crook': 705, 'Jimmy Stewart': 706, 'Ian Stewart': 707, 'Duncan Hamilton': 708, 'Ernst Klodwig': 709, 'Rudolf Krause': 710, 'Oswald Karch': 711, 'Willi Heeks': 712, 'Theo Fitzau': 713, 'Kurt Adolff': 714, 'Günther Bechem': 715, 'Erwin Bauer': 716, 'Hans von Stuck': 717, 'Ernst Loof': 718, 'Albert Scherrer': 719, 'Max de Terra': 720, 'Peter Hirt': 721, 'Piero Carini': 722, 'Rudi Fischer': 723, 'Toni Ulmen': 724, 'George Abecassis': 725, 'George Connor': 726, 'Jim Rigsby': 727, 'Joe James': 728, 'Bill Schindler': 729, 'George Fonder': 730, 'Henry Banks': 731, 'Johnny McDowell': 732, 'Chet Miller': 733, 'Bobby Ball': 734, 'Charles de Tornaco': 735, 'Roger Laurent': 736, "Robert O'Brien": 737, 'Tony Gaze': 738, 'Robin Montgomerie-Charrington': 739, 'Franco Comotti': 740, 'Philippe Étancelin': 741, 'Dennis Poore': 742, 'Eric Thompson': 743, 'Ken Downing': 744, 'Graham Whitehead': 745, 'Gino Bianco': 746, 'David Murray': 747, 'Eitel Cantoni': 748, 'Bill Aston': 749, 'Adolf Brudes': 750, 'Fritz Riess': 751, 'Helmut Niedermayr': 752, 'Hans Klenk': 753, 'Marcel Balsa': 754, 'Rudolf Schoeller': 755, 'Paul Pietsch': 756, 'Josef Peters': 757, 'Dries van der Lof': 758, 'Jan Flinterman': 759, 'Piero Dusio': 760, 'Alberto Crespo': 761, 'Franco Rol': 762, 'Consalvo Sanesi': 763, 'Guy Mairesse': 764, 'Henri Louveau': 765, 'Lee Wallard': 766, 'Carl Forberg': 767, 'Mauri Rose': 768, 'Bill Mackey': 769, 'Cecil Green': 770, 'Walt Brown': 771, 'Mack Hellings': 772, 'Pierre Levegh': 773, 'Eugène Chaboud': 774, 'Aldo Gordini': 775, 'Joe Kelly': 776, 'Philip Fotheringham-Parker': 777, 'Brian Shawe Taylor': 778, 'John James': 779, 'Toni Branca': 780, 'Ken Richardson': 781, 'Juan Jover': 782, 'Georges Grignard': 783, 'David Hampshire': 784, 'Geoff Crossley': 785, 'Luigi Fagioli': 786, 'Cuth Harrison': 787, 'Joe Fry': 788, 'Eugène Martin': 789, 'Leslie Johnson': 790, 'Clemente Biondetti': 791, 'Alfredo Pián': 792, 'Raymond Sommer': 793, 'Joie Chitwood': 794, 'Myron Fohr': 795, 'Walt Ader': 796, 'Jackie Holmes': 797, 'Bayliss Levrett': 798, 'Jimmy Jackson': 799, 'Nello Pagani': 800, 'Charles Pozzi': 801, 'Dorino Serafini': 802, 'Bill Cantrell': 803, 'Johnny Mantz': 804, 'Danny Kladis': 805, 'Óscar González': 806, 'Nico Hülkenberg': 807, 'Vitaly Petrov': 808, 'Lucas di Grassi': 810, 'Bruno Senna': 811, 'Karun Chandhok': 812, 'Pastor Maldonado': 813, 'Paul di Resta': 814, 'Sergio Pérez': 815, "Jérôme d'Ambrosio": 816, 'Daniel Ricciardo': 817, 'Jean-Éric Vergne': 818, 'Charles Pic': 819, 'Max Chilton': 820, 'Esteban Gutiérrez': 821, 'Valtteri Bottas': 822, 'Giedo van der Garde': 823, 'Jules Bianchi': 824, 'Kevin Magnussen': 825, 'Daniil Kvyat': 826, 'André Lotterer': 827, 'Marcus Ericsson': 828, 'Will Stevens': 829, 'Max Verstappen': 830, 'Felipe Nasr': 831, 'Carlos Sainz': 832, 'Roberto Merhi': 833, 'Alexander Rossi': 834, 'Jolyon Palmer': 835, 'Pascal Wehrlein': 836, 'Rio Haryanto': 837, 'Stoffel Vandoorne': 838, 'Esteban Ocon': 839, 'Lance Stroll': 840, 'Antonio Giovinazzi': 841, 'Brendon Hartley': 843, 'Charles Leclerc': 844, 'Sergey Sirotkin': 845, 'Lando Norris': 846, 'George Russell': 847, 'Alexander Albon': 848, 'Nicholas Latifi': 849, 'Pietro Fittipaldi': 850, 'Jack Aitken': 851, 'Yuki Tsunoda': 852, 'Nikita Mazepin': 853, 'Mick Schumacher': 854, 'Guanyu Zhou': 855, 'Nyck de Vries': 856, 'Oscar Piastri': 857, 'Logan Sargeant': 858, 'Liam Lawson': 859, 'Oliver Bearman': 860, 'Franco Colapinto': 861, 'Jack Doohan': 862}
constructors= {'McLaren': 1, 'BMW Sauber': 2, 'Williams': 3, 'Renault': 4, 'Toro Rosso': 5, 'Ferrari': 6, 'Toyota': 7, 'Super Aguri': 8, 'Red Bull': 9, 'Force India': 10, 'Honda': 11, 'Spyker': 12, 'MF1': 13, 'Spyker MF1': 14, 'Sauber': 15, 'BAR': 16, 'Jordan': 17, 'Minardi': 18, 'Jaguar': 19, 'Prost': 20, 'Arrows': 21, 'Benetton': 22, 'Brawn': 23, 'Stewart': 24, 'Tyrrell': 25, 'Lola': 26, 'Ligier': 27, 'Forti': 28, 'Footwork': 29, 'Pacific': 30, 'Simtek': 31, 'Team Lotus': 32, 'Larrousse': 33, 'Brabham': 34, 'Dallara': 35, 'Fondmetal': 36, 'March': 37, 'Andrea Moda': 38, 'AGS': 39, 'Lambo': 40, 'Leyton House': 41, 'Coloni': 42, 'Euro Brun': 44, 'Osella': 45, 'Onyx': 46, 'Life': 47, 'Rial': 48, 'Zakspeed': 49, 'RAM': 50, 'Alfa Romeo': 51, 'Spirit': 52, 'Toleman': 53, 'ATS': 54, 'Theodore': 55, 'Fittipaldi': 56, 'Ensign': 57, 'Shadow': 58, 'Wolf': 59, 'Merzario': 60, 'Kauhsen': 61, 'Rebaque': 62, 'Surtees': 63, 'Hesketh': 64, 'Martini': 65, 'BRM': 66, 'Penske': 67, 'LEC': 68, 'McGuire': 69, 'Boro': 70, 'Apollon': 71, 'Kojima': 72, 'Parnelli': 73, 'Maki': 74, 'Embassy Hill': 75, 'Lyncar': 76, 'Trojan': 77, 'Amon': 78, 'Token': 79, 'Iso Marlboro': 80, 'Tecno': 81, 'Matra': 82, 'Politoys': 83, 'Connew': 84, 'Bellasi': 85, 'De Tomaso': 86, 'Cooper': 87, 'Eagle': 88, 'LDS': 89, 'Protos': 90, 'Shannon': 91, 'Scirocco': 92, 'RE': 93, 'BRP': 94, 'Porsche': 95, 'Derrington': 96, 'Gilby': 97, 'Stebro': 98, 'Emeryson': 99, 'ENB': 100, 'JBW': 101, 'Ferguson': 102, 'MBM': 103, 'Behra-Porsche': 104, 'Maserati': 105, 'Scarab': 106, 'Watson': 107, 'Epperly': 108, 'Phillips': 109, 'Lesovsky': 110, 'Trevis': 111, 'Meskowski': 112, 'Kurtis Kraft': 113, 'Kuzma': 114, 'Christensen': 115, 'Ewing': 116, 'Aston Martin': 117, 'Vanwall': 118, 'Moore': 119, 'Dunn': 120, 'Elder': 121, 'Sutton': 122, 'Fry': 123, 'Tec-Mec': 124, 'Connaught': 125, 'Alta': 126, 'OSCA': 127, 'Gordini': 128, 'Stevens': 129, 'Bugatti': 130, 'Mercedes': 131, 'Lancia': 132, 'HWM': 133, 'Schroeder': 134, 'Pawl': 135, 'Pankratz': 136, 'Arzani-Volpini': 137, 'Nichels': 138, 'Bromme': 139, 'Klenk': 140, 'Simca': 141, 'Turner': 142, 'Del Roy': 143, 'Veritas': 144, 'BMW': 145, 'EMW': 146, 'AFM': 147, 'Frazer Nash': 148, 'Sherman': 149, 'Deidt': 150, 'ERA': 151, 'Aston Butterworth': 152, 'Cisitalia': 153, 'Talbot-Lago': 154, 'Hall': 155, 'Marchese': 156, 'Langley': 157, 'Rae': 158, 'Olson': 159, 'Wetteroth': 160, 'Adams': 161, 'Snowberger': 162, 'Milano': 163, 'HRT': 164, 'Cooper-Maserati': 167, 'Virgin': 166, 'Cooper-OSCA': 168, 'Cooper-Borgward': 169, 'Cooper-Climax': 170, 'Cooper-Castellotti': 171, 'Lotus-Climax': 172, 'Lotus-Maserati': 173, 'De Tomaso-Osca': 174, 'De Tomaso-Alfa Romeo': 175, 'Lotus-BRM': 176, 'Lotus-Borgward': 177, 'Cooper-Alfa Romeo': 178, 'De Tomaso-Ferrari': 179, 'Lotus-Ford': 180, 'Brabham-BRM': 181, 'Brabham-Ford': 182, 'Brabham-Climax': 183, 'LDS-Climax': 184, 'LDS-Alfa Romeo': 185, 'Cooper-Ford': 186, 'McLaren-Ford': 187, 'McLaren-Serenissima': 188, 'Eagle-Climax': 189, 'Eagle-Weslake': 190, 'Brabham-Repco': 191, 'Cooper-Ferrari': 192, 'Cooper-ATS': 193, 'McLaren-BRM': 194, 'Cooper-BRM': 195, 'Matra-Ford': 196, 'BRM-Ford': 197, 'McLaren-Alfa Romeo': 198, 'March-Alfa Romeo': 199, 'March-Ford': 200, 'Lotus-Pratt &amp; Whitney': 201, 'Shadow-Ford': 202, 'Shadow-Matra': 203, 'Brabham-Alfa Romeo': 204, 'Lotus': 205, 'Marussia': 206, 'Caterham': 207, 'Lotus F1': 208, 'Manor Marussia': 209, 'Haas F1 Team': 210, 'Racing Point': 211, 'AlphaTauri': 213, 'Alpine F1 Team': 214, 'RB F1 Team': 215}
circuits= {'Melbourne, Australia': 1, 'Kuala Lumpur, Malaysia': 2, 'Sakhir, Bahrain': 3, 'Montmeló, Spain': 4, 'Istanbul, Turkey': 5, 'Monte-Carlo, Monaco': 6, 'Montreal, Canada': 7, 'Magny Cours, France': 8, 'Silverstone, UK': 9, 'Hockenheim, Germany': 10, 'Budapest, Hungary': 11, 'Valencia, Spain': 12, 'Spa, Belgium': 13, 'Monza, Italy': 14, 'Marina Bay, Singapore': 15, 'Oyama, Japan': 16, 'Shanghai, China': 17, 'São Paulo, Brazil': 18, 'Indianapolis, USA': 19, 'Nürburg, Germany': 20, 'Imola, Italy': 21, 'Suzuka, Japan': 22, 'Las Vegas, United States': 80, 'Abu Dhabi, UAE': 24, 'Buenos Aires, Argentina': 25, 'Jerez de la Frontera, Spain': 26, 'Estoril, Portugal': 27, 'Okayama, Japan': 28, 'Adelaide, Australia': 29, 'Midrand, South Africa': 30, 'Castle Donington, UK': 31, 'Mexico City, Mexico': 32, 'Phoenix, USA': 33, 'Le Castellet, France': 34, 'Yeongam County, Korea': 35, 'Rio de Janeiro, Brazil': 36, 'Detroit, USA': 37, 'Kent, UK': 38, 'Zandvoort, Netherlands': 39, 'Heusden-Zolder, Belgium': 40, 'Dijon, France': 41, 'Dallas, USA': 42, 'California, USA': 60, 'Nevada, USA': 44, 'Madrid, Spain': 45, 'New York State, USA': 46, 'Anderstorp, Sweden': 47, 'Ontario, Canada': 48, 'Barcelona, Spain': 67, 'Brussels, Belgium': 50, 'Clermont-Ferrand, France': 51, 'Quebec, Canada': 52, 'Rouen, France': 53, 'Le Mans, France': 54, 'Reims, France': 55, 'Eastern Cape Province, South Africa': 56, 'Styria, Austria': 57, 'Liverpool, UK': 58, 'Oporto, Portugal': 59, 'Berlin, Germany': 61, 'Lisbon, Portugal': 62, 'Florida, USA': 63, 'Casablanca, Morocco': 64, 'Pescara, Italy': 65, 'Bern, Switzerland': 66, 'Uttar Pradesh, India': 68, 'Austin, USA': 69, 'Spielberg, Austria': 70, 'Sochi, Russia': 71, 'Baku, Azerbaijan': 73, 'Portimão, Portugal': 75, 'Mugello, Italy': 76, 'Jeddah, Saudi Arabia': 77, 'Al Daayen, Qatar': 78, 'Miami, USA': 79}


# -----------------------
# UI
# -----------------------

st.title("🏎️ F1 Race Position Predictor")

st.write("Select race parameters and predict finishing position")

tab1, tab3 = st.tabs([
    "Single Prediction",
    "Bulk Prediction (Paste CSV)"
])

def find_best_match(text, dictionary):

    if text in dictionary:
        return dictionary[text]

    matches = difflib.get_close_matches(text, dictionary.keys(), n=1, cutoff=0.6)

    if matches:
        return dictionary[matches[0]]

    return 0

# =========================================================
# TAB 1 — EXISTING SINGLE PREDICTION (UNCHANGED)
# =========================================================

with tab1:

    st.write("Select race parameters and predict finishing position")

    grid = st.slider("Grid Position", 1, 20)

    driver_name = st.selectbox(
        "Search Driver",
        sorted(drivers.keys())
    )

    constructor_name = st.selectbox(
        "Search Constructor",
        sorted(constructors.keys())
    )

    circuit_name = st.selectbox(
        "Search Circuit",
        sorted(circuits.keys())
    )

    year = st.selectbox(
        "Year",
        list(range(2000, 2027))
    )

    driver_id = drivers[driver_name]
    constructor_id = constructors[constructor_name]
    circuit_id = circuits[circuit_name]

    if st.button("Predict Result"):

        features = np.array([[grid, driver_id, constructor_id, circuit_id, year]])

        prediction = model.predict(features)

        st.success(f"🏁 Predicted Position: {int(prediction[0])}")


# =========================================================
# TAB 2 — CSV BULK RACE PREDICTION
# =========================================================

# with tab2:

#     st.subheader("Upload Race Grid CSV")

#     st.write("Upload a CSV like:")

#     st.code(
# """
# Grid,Driver,Team
# 1,George Russell,Mercedes
# 2,Kimi Antonelli,Mercedes
# 3,Charles Leclerc,Ferrari
# """
# )
    
#     # Circuit and Year selectors for bulk prediction
#     circuit_name_bulk = st.selectbox(
#         "Select Circuit",
#         sorted(circuits.keys()),
#         key="bulk_circuit"
#     )
    
#     year_bulk = st.selectbox(
#         "Select Year",
#         list(range(2000, 2027)),
#         key="bulk_year"
#     )

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

#     if uploaded_file:

#         df = pd.read_csv(uploaded_file)

#         st.dataframe(df)

#         predictions = []

#         for _, row in df.iterrows():

#             grid_val = int(row["Grid"])
#             driver_name = row["Driver"]
#             team_name = row["Team"]

#             driver_id = find_best_match(driver_name, drivers)
#             constructor_id = find_best_match(team_name, constructors)

#             features = np.array([
#                 [grid_val, driver_id, constructor_id, circuit_id_bulk, year_bulk]
#             ])

#             pred = model.predict(features)[0]

#             predictions.append(pred)

#         df["Predicted_Position"] = predictions

#         # ensure unique positions
#         df = df.sort_values("Predicted_Position")
#         df["Predicted_Position"] = range(1, len(df)+1)

#         st.subheader("🏁 Predicted Race Result")

#         st.dataframe(
#             df[["Predicted_Position","Grid","Driver","Team"]]
#         )
        

# =========================================================
# TAB 3 — PASTE CSV TEXT
# =========================================================

with tab3:

    st.subheader("Paste CSV Data")

    csv_text = st.text_area("Paste CSV")
    
    # Circuit and Year selectors for bulk prediction
    circuit_name_bulk = st.selectbox(
        "Select Circuit",
        sorted(circuits.keys()),
        key="bulk_circuit"
    )

    year_bulk = st.selectbox(
        "Select Year",
        list(range(2000, 2027)),
        key="bulk_year"
    )

circuit_id_bulk = circuits[circuit_name_bulk]

if st.button("Predict From Pasted CSV"):

    if csv_text.strip():

        df = pd.read_csv(StringIO(csv_text))

        predictions = []
        

        for _, row in df.iterrows():

            grid_val = int(row["Grid"])
            driver_name = row["Driver"]
            team_name = row["Team"]

            driver_id = find_best_match(driver_name, drivers)
            constructor_id = find_best_match(team_name, constructors)

            features = np.array([
                [grid_val, driver_id, constructor_id, circuit_id_bulk, year_bulk]
            ])

            pred = model.predict(features)[0]

            predictions.append(pred)

        df["Predicted_Position"] = predictions

        df = df.sort_values("Predicted_Position")
        df["Predicted_Position"] = range(1, len(df)+1)

        df = df.reset_index(drop=True)

        st.dataframe(df, hide_index=True)