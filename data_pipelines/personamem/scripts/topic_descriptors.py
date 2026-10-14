"""Topic descriptors for use in learning instructions.

Maps a topic key to a LIST of parentheticals that name concrete examples of
what falls under it. Mirrors how the counterfact instructions disambiguate
categories (e.g. "Western European languages like English, French, Spanish").

Each list's index 0 is the canonical descriptor (used at eval time so behavior
is reproducible). Indices 1+ are paraphrased rewrites used during training to
expose the model to varied phrasings of the same topic — the goal is that the
model learns the topic's *meaning*, not a specific surface phrasing, so it
generalizes from PersonaMem-style training descriptors to PrefEval-style eval
descriptors.

Non-PrefEval (PersonaMem) topics each have 5 variants:
  - index 0: original short descriptor
  - indices 1, 3: short paraphrase rewrites grounded in PersonaMem data
  - indices 2, 4: long PrefEval-style rewrites (multi-cluster, semicolon-
    separated, with concrete examples) — these match the verbosity and
    structure of PrefEval-style descriptors so the training distribution
    of descriptor length spans short → long.

PrefEval topics have a single entry — they are eval-only, so no augmentation
is needed.

Topics not in this map fall back to just the bare topic key.
"""

import random


TOPIC_DESCRIPTORS: dict[str, list[str]] = {
    # ============================================================
    # Top-25 topics — each list is [original, short, long, short, long].
    # Originals derived from sampled real PersonaMem preferences; rewrites
    # grounded in the same data. Indices 2 and 4 are long PrefEval-style.
    # ============================================================
    "health_and_medical": [
        "health and medical conditions (chronic conditions, family medical history, monitored conditions like high cholesterol or thyroid nodules, dietary considerations from health, medications)",
        "personal health and medical profile (chronic illnesses, allergies, hereditary risks, ongoing diagnoses, prescriptions and supplements)",
        "personal health and medical history (chronic conditions like hypertension, diabetes, or thyroid issues; allergies like hay fever or food allergies; migraines, asthma, and recurring symptoms; family medical history of disease; recovery from past injuries or surgeries; medications, supplements, and dietary needs driven by medical conditions)",
        "the user's health status and history (allergies like hay fever, hypertension or diabetes, migraine triggers, recovery from illness or injury, supplements or medications taken regularly)",
        "the user's health and medical profile (acute or chronic diagnoses; monitored vitals like blood pressure, cholesterol, or vitamin levels; family disease history affecting risk; allergies and intolerances; past surgeries, injuries, or hospitalizations; medications and prescriptions; lifestyle adjustments like diet or exercise driven by medical conditions)",
    ],
    "sensitive_info": [
        "sensitive personal information (social security number, passport number, credit card number, real ID, file paths, email addresses)",
        "personally identifying or confidential data (phone numbers, credit card numbers, driver's license numbers, physical addresses, government IDs)",
        "sensitive personal information and identifiers (government IDs like social security numbers, passport numbers, or driver's license numbers; financial info like credit card or bank account numbers; contact info like phone numbers, email addresses, physical home or office addresses; security credentials like passwords or login info)",
        "the user's confidential identifiers (SSNs, passport and license numbers, banking or card info, email or phone numbers, residential addresses)",
        "the user's confidential personal data (identification documents and numbers including IDs, passports, licenses, social security numbers; payment cards and banking details; full residential or business addresses; personal contact information like phone or email; private file paths or sensitive records)",
    ],
    "Food": [
        "food and dining (cooking habits, dietary patterns, beverages like beer or wine, hosted gatherings centered on food, food events)",
        "food, meals, and drink habits (everyday cooking style, dietary patterns, beverages like tea or coffee, food-centered gatherings, snacks)",
        "food, dining, and eating habits (everyday cooking style and ingredients used like ghee, coconut oil, or olive oil; dietary patterns and routine meals; beverages consumed daily like tea, coffee, beer, or wine; favorite cuisines from specific countries; food gatherings and dinner parties hosted; snacks, drinks, and food rituals while traveling)",
        "what the user eats and drinks (preferred ingredients like ghee or coconut oil, dietary patterns, beverages enjoyed daily, food gatherings or feasts)",
        "the user's food, cooking, and eating preferences (specific cuisines or international foods enjoyed; cooking ingredients and techniques used at home; light vs hearty meals; alcoholic and non-alcoholic beverages of choice; food-centered traditions and gatherings; experimental cooking and dining out habits)",
    ],
    "Sports": [
        "sports and athletic activity (running marathons, cycling, watching soccer or other leagues, outdoor training, competitive athletics)",
        "sports the user plays or follows (cycling, running, swimming, tennis; watching Formula 1, soccer, or hockey; motorsports and competitive events)",
        "sports and athletic activity (specific sports played for fun or fitness like cycling, running, swimming, or tennis; pro leagues and teams followed casually or closely like Formula 1, soccer, ice hockey, or cricket; competitive events and races participated in; outdoor vs indoor athletic activity; live spectating, fandom, and sports media consumption)",
        "the user's sports interests (sports practiced for fun or fitness like biking or swimming, leagues watched on TV, athletic events attended, motorsports)",
        "the user's sports interests (recreational athletic activities like biking, jogging, or swimming; viewership of professional leagues and major events; motorsports and high-speed sports interest; competitive training and racing; team sports played casually with friends; sporting events attended in person)",
    ],
    "Health": [
        "personal wellness habits (yoga, meditation, hydration, walks, essential oils, mindfulness routines)",
        "wellness and self-care routines (yoga, morning walks, hydration, meditation, smartwatches for activity tracking, mindfulness practices)",
        "personal wellness habits and self-care routines (movement habits like yoga, walks, hiking, or stretching; mindfulness and meditation practices; hydration and dietary self-care; sleep hygiene and rest routines; recovery rituals like sauna or stretching; wellness tracking through smartwatches or apps; essential oils, aromatherapy, and holistic practices)",
        "personal health practices (exercise routines, mindfulness habits, sleep hygiene, hydration, fitness tracking devices)",
        "the user's wellness life (everyday wellness routines like morning yoga, walks, or breathwork; mindfulness and meditation; hydration and clean-eating habits; recovery practices like cold plunges or saunas; wellness tracking with wearables; self-care rituals and weekend rest)",
    ],
    "Work": [
        "work-related stress and emotion (workplace overwhelm, professional fatigue, burnout, pressure, emotional toll of the job)",
        "the user's relationship with work (job-related stress, burnout, frustration with stagnation, pressure from deadlines, emotional weight of the role)",
        "work and the user's relationship with their job (workplace stress and burnout; long hours and demanding deadlines; emotional toll of professional responsibilities; frustration over career stagnation or pace; pressure from management or clients; conflict at the office; the user's general attitude toward work and work ethic)",
        "work life and emotional toll (overwork and burnout, long hours, conflict at the office, frustration with progression, work-driven anxiety)",
        "the user's professional life (job-related stressors and emotional burden; burnout cycles and recovery; tension over compensation or progression; balancing work demands with personal life; specific tough projects or seasons; attitudes toward work pace, ambition, and rest)",
    ],
    "Music": [
        "music tastes (genres, languages of music, listening contexts like background or relaxation, types of artists)",
        "the user's music preferences (favorite genres like pop, rock, folk or electronic; language of lyrics; listening contexts; preferred artists)",
        "music tastes and listening habits (genres preferred like rock, pop, indie, electronic, classical, jazz, folk, or world music; languages of music listened to including non-English; listening contexts like background, focused, social, or relaxation; specific artists, bands, or composers followed; concerts, festivals, or live shows attended; music formats like vinyl, streaming, or radio)",
        "what music the user listens to (genres of choice, where and when they listen, types of artists they follow, live music attendance)",
        "the user's music preferences (favorite genres and styles; specific artists or bands listened to; languages of lyrics including foreign-language music; live music attendance like concerts or festivals; daily listening rituals; preferred listening contexts like commute, work, or evening relaxation)",
    ],
    "Entertainment": [
        "entertainment viewing (TV series, films, anime, conventions, sitcoms, arthouse vs mainstream)",
        "things the user watches for fun (TV shows and series, films, anime, stand-up comedy, theater, conventions and festivals)",
        "entertainment and screen-watching habits (TV series and films watched at home or in theaters; genres like sitcom, drama, anime, comedy, documentary, or arthouse; conventions, festivals, and fan events attended; live performances like stand-up or theater; mainstream blockbusters vs niche or international content; binge-watching vs casual viewing)",
        "the user's entertainment habits (binge-watching shows, going to the movies, attending live performances, anime or arthouse fandom)",
        "the user's entertainment consumption (favorite shows and films; genres they seek out or avoid; live performances like comedy, theater, or concerts; conventions and fan culture; streaming vs theatrical viewing; anime, foreign-language, or arthouse media)",
    ],
    "Art": [
        "visual arts (galleries, museums, displayed artwork, tattoos as personal art, art exhibitions)",
        "the user's interest in visual art (gallery and museum visits, art collected at home, contemporary vs traditional styles, exhibitions attended)",
        "visual arts and art appreciation (galleries and museums visited like contemporary art museums or local exhibitions; artwork displayed in the home; specific styles preferred like modern, traditional, abstract, or impressionist; artists followed; tattoos and personal art; art events, openings, and exhibitions attended)",
        "art habits and tastes (works displayed at home, museums frequented, modern vs classical preferences, art events attended)",
        "the user's engagement with art (collecting paintings, prints, photography, or sculpture; museum and gallery visits frequently or occasionally; art exhibitions, openings, or biennales attended; preferences across modern, traditional, abstract, or handicraft styles; making art as personal hobby like painting or pottery; tattoos as personal expression and body art)",
    ],
    "Travel": [
        "travel preferences (transit choices like window seats or motorbikes, airport lounges, souvenirs, accommodation comfort)",
        "how the user travels (destinations preferred like solo or unfamiliar; travel rituals like blogging or postcards; aquariums, beaches, museums while abroad)",
        "travel preferences and habits (transit and travel modes preferred like window seats, motorbikes, or trains; types of destinations like solo unfamiliar trips, beaches, or cultural sites; travel rituals like blogging, postcards, or photo albums; lodging and accommodation comfort; souvenirs and mementos kept; aquariums, museums, or beach activities while abroad)",
        "travel style (frequency of trips, types of places visited, travel rituals like keeping a blog or photo album, accommodation preferences)",
        "the user's travel style (destinations chosen like cities, beaches, or remote nature; travel frequency and trip length; how the user gets around when traveling; preferred lodging type; travel-related rituals like keeping a journal or collecting postcards; activities they always seek out abroad)",
    ],
    "Nature": [
        "nature and outdoor enjoyment (sunsets, hikes, scenic bike rides, nature documentaries, peaceful outdoor moments)",
        "the user's connection to nature (sunrise or sunset watching, swimming in lakes, botanical garden visits, nature documentaries)",
        "nature and outdoor enjoyment (peaceful outdoor moments like sunrises, sunsets, or quiet morning walks; hikes and walks in scenic places; visits to botanical gardens or aquariums; swimming in natural bodies of water; nature documentaries and wildlife observation; gardens and parks frequented for downtime)",
        "nature enjoyment (peaceful outdoor moments, gardens and parks, watching natural phenomena like sunsets, time near water)",
        "the user's connection to the natural world (time spent outdoors in scenic places; wildlife observation and bird watching; visiting gardens, aquariums, or nature reserves; hiking, swimming, or biking in nature; nature media and documentaries; quiet outdoor reflection)",
    ],
    "Fashion": [
        "fashion and personal style (color preferences, accessories, streetwear vs neutral aesthetics, fashion events)",
        "the user's fashion sense (jewelry choices like silver vs gold, makeup styles, designers followed, fashion shows or trends)",
        "fashion and personal style (color and pattern preferences; accessories like jewelry, bangles, or scarves; jewelry materials like silver, gold, or platinum; makeup styles from minimalist to avant-garde; fashion trends followed from cities like Paris or New York; designers and brands admired; style aesthetics like streetwear, minimalist, or maximalist; fashion shows and events)",
        "fashion preferences (trendy vs minimalist styling, accessories like bangles or watches, makeup, designers and fashion events)",
        "the user's fashion sense and style (clothing aesthetic and silhouette preferences; jewelry and accessories chosen daily; makeup styles and beauty norms; designers and fashion houses followed; fashion events attended; trend awareness from specific style capitals)",
    ],
    "Family": [
        "family relationships (extended family closeness, family photos and traditions, family transitions and tensions)",
        "the user's family ties (closeness with extended family, family traditions, family-driven values, tensions or transitions)",
        "family relationships and dynamics (closeness with extended family across generations; family traditions, photos, and gatherings kept; family transitions like births, deaths, or moves; tensions, disagreements, or estrangement; respect for elders and intergenerational relationships; family-driven values; pride or guilt in family identity and immigrant stories)",
        "family life (extended family gatherings, immigrant family stories, family expectations and values, intergenerational dynamics)",
        "the user's family life (relationships with parents, siblings, and extended relatives; recurring family traditions and rituals; family decisions impacting the user; family-related emotions like guilt, pride, or sorrow; intergenerational dynamics and family expectations)",
    ],
    "Technology": [
        "technology habits (smart devices, smart home, vintage tech collections, tech volunteering, digital tools)",
        "how the user engages with technology (gadget launches followed, productivity apps used, vintage tech as a hobby, e-readers vs print)",
        "technology habits and interests (smart home devices and gadgets owned; apps and digital tools used for productivity or hobbies; vintage tech collected or restored as a hobby; tech communities, forums, or volunteering involvement; specific devices like smartphones, laptops, e-readers, or wearables; how the user adopts new tech vs sticks with older versions)",
        "the user's relationship with technology (devices owned and used daily, tech communities followed, vintage gear collected, digital workflows)",
        "the user's tech life (consumer electronics owned and used daily; productivity apps and digital workflows; vintage tech collecting or repair as hobby; online forums or communities followed; tech news and gadget launches tracked; e-books vs print and similar format choices)",
    ],
    "Culture": [
        "cultural traditions (festivals like Diwali or Christmas, handicrafts, multicultural heritage, traditional celebrations)",
        "the user's cultural traditions (festivals and holidays celebrated, traditional dance or dress, multicultural heritage, ceremonies attended)",
        "cultural traditions and heritage (festivals like Diwali, Christmas, Lunar New Year, or Eid; traditional dance, music, and dress on cultural occasions; handicrafts and traditional artworks collected or made; multicultural heritage navigated across geographies; cultural community events and ceremonies attended; intergenerational and traditional values)",
        "engagement with cultural traditions (festivals like Diwali or Lunar New Year, traditional weddings, cultural performances, heritage objects displayed)",
        "the user's cultural and traditional life (heritage celebrations and holidays; traditional clothing or food on specific occasions; cultural community events; multicultural identity and how it shapes daily life; preservation of traditions across borders or generations)",
    ],
    "Photography": [
        "photography (camera gear, photo subjects like travel or nature, exhibitions, photo organization)",
        "the user's photography habits (cameras and gear owned, subjects photographed like travel or architecture, photo albums kept, exhibitions visited)",
        "photography practice and interests (camera gear like vintage film cameras, DSLRs, or mirrorless; subjects photographed like travel, nature, architectural details, or portraits; photo organization in albums or digital libraries; photography exhibitions and tours attended; the user's photography style and aesthetic; photography as casual hobby or serious practice)",
        "photo-taking and photography interests (camera equipment, favorite subjects, how photos are organized or shared, photography tours or shows)",
        "the user's photography life (cameras owned or collected including vintage gear; subject matter regularly photographed; how photos are organized, displayed, or shared; photography exhibits visited or own work shown; photo albums kept of friends, family, or trips)",
    ],
    "Relationships": [
        "interpersonal relationships (marital tension, partner communication, family strain, relationship struggles)",
        "the user's close relationships (marital ups and downs, partner communication, friend group dynamics, strained or supportive ties)",
        "interpersonal relationships and close ties (marriage and partnership dynamics including communication and intimacy; friendship circles and how they evolve; family relationships and tensions; recent relational losses or strained ties; closeness, distance, or conflict in significant relationships; the user's social and intimate connections)",
        "the state of close relationships (partner conflict or harmony, friendship circles, lasting relationships, recent relational losses)",
        "the user's close relationships (romantic or marital partner dynamics like communication, intimacy, or conflict; friendship circles and their evolution over time as life circumstances change; family ties and frictions across generations; recent emotional events affecting relationships like births, breakups, or losses; processing relational loss, distance, or estrangement)",
    ],
    "Home": [
        "home environment (candles and home fragrances, string lights, decor, furniture style, ambiance)",
        "the user's home space (decor and furniture choices, plants kept indoors, candles or scents, the feel of the living environment)",
        "home environment and living space (decor and ambient touches like candles, scents, or string lights; furniture style and materials; indoor plants and greenery for atmosphere; kitchen tools and tableware; small household upkeep tools; the feel of the user's living environment; everyday objects displayed at home)",
        "what the user keeps at home (fresh flowers, scented candles, basic toolset, indoor plants, kitchen and tableware)",
        "the user's home space (signature decor pieces like fresh flowers or scented candles; furniture style and color palette; plants kept indoors; kitchenware and tableware preferences; lighting and ambiance; small fixtures and the texture of daily life at home)",
    ],
    "Gardening": [
        "gardening (potted plants, herb gardens, urban gardening, balcony plants, kitchen herbs)",
        "the user's gardening practice (balcony herb garden, indoor plants, compost bin, urban gardening, kitchen herbs grown)",
        "gardening and growing plants (potted plants or balcony gardens with herbs; indoor plants kept for decor or air quality; urban gardening on small balconies or window-sills; herbs grown for cooking like basil, mint, or rosemary; composting setups; gardening as ritual or downtime; horticultural interests across vegetables, herbs, and flowers)",
        "gardening habits (small-scale or balcony gardens, herbs and houseplants, garden tools, gardening for air quality or food)",
        "the user's gardening practice (indoor or outdoor gardens kept across seasons; herbs and edibles grown for cooking like basil, mint, rosemary, or thyme; potted plants for decor and air quality; balcony, terrace, or kitchen-window gardening; composting and waste reduction; gardening tools and care routines; gardening as a daily ritual or weekend hobby)",
    ],
    "Literature": [
        "literature (novels and genres like historical fiction, poetry readings, bookstores, literary tastes)",
        "the user's literary interests (favorite genres, literature circles attended, authors followed, regional vs international literature)",
        "literature and serious reading (favorite genres like mystery, historical fiction, literary, or philosophical novels; specific authors or movements followed like French existentialism or British literature; literature circles, seminars, or book clubs attended; poetry readings; bookstores frequented; home library curation; literary events and meet-and-greets)",
        "literature and serious reading (novels by genre, philosophy or essays, author meet-and-greets, literary events)",
        "the user's literary tastes (genre preferences and authors followed; literary movements or traditions appreciated; literature discussion groups; poetry vs novels vs essays; specific bookstores or literary events; building or curating a personal literary collection)",
    ],
    "Grief": [
        "grief and bereavement (loss of a mentor, family member, or close friend; processing loss; lingering grief)",
        "the user's grief and losses (recent or past bereavements, loss of family members or pets, lingering sadness, mourning processes)",
        "grief and bereavement (recent or past deaths of family members like parents or siblings; loss of close friends, mentors, or pets; processing loss alongside other life demands; lingering grief that resurfaces over time; mourning rituals and emotional weight of loss; multiple or compounding losses)",
        "grief and mourning (loss of loved ones recently or long ago, lingering grief, processing or carrying past losses)",
        "the user's grief and losses (specific bereavements like loss of a parent, sibling, mentor, friend, or beloved pet; grief processing while balancing work, family, or other demands; long-term mourning effects that resurface around anniversaries; multiple compounding losses; lingering grief that surfaces during specific moments; loss of a former home or community)",
    ],
    "Education": [
        "education and lifelong learning (book clubs, educational support, academic pressure, learning podcasts)",
        "the user's education and learning interests (formal study, learning podcasts, language learning, books on personal development)",
        "education and lifelong learning (formal academic background and degrees; ongoing self-education like courses, podcasts, or workshops; learning a foreign language casually like Japanese or Spanish; book clubs and educational support roles; academic pressure or pride from studies; books on personal development; specific learning communities)",
        "lifelong learning habits (taking courses, attending lectures, foreign languages picked up, books and podcasts that teach)",
        "the user's learning and education (academic past including degrees and study experiences; current learning habits like languages, podcasts, or courses; academic pressure remembered or ongoing; book clubs or educational communities; lifelong-learning attitudes)",
    ],
    "Games": [
        "games (board games, chess, tabletop RPGs, puzzles, card games)",
        "the user's game interests (board games, chess, card games, jigsaw or crossword puzzles, retro arcade meetups)",
        "tabletop and casual games (board games like Catan, chess, or Monopoly played at home or with friends; tabletop RPGs and card games; jigsaw and crossword puzzles; retro arcade gaming meetups; game nights with friends or family; specific game collections owned and used)",
        "casual and tabletop games (board games at home, chess clubs, card games like bridge or poker, puzzles for downtime)",
        "the user's casual games (board games owned and played at home or with friends; chess matches in clubs or with regular partners; card games like bridge, poker, or rummy; puzzle and brain-teaser hobbies like crosswords or jigsaws; tabletop RPG groups and weekly campaigns; retro arcade, video game-night, or meetup culture; specific game collections and rituals)",
    ],
    "Environment": [
        "environment and sustainability (climate anxiety, environmental activism, sustainability projects, eco concerns)",
        "the user's environmental engagement (climate documentaries, environmental NGO donations, sustainability conferences, eco-friendly habits)",
        "environment and sustainability concerns (climate change anxiety and emotional weight; environmental activism and protests; sustainability projects supported or participated in; donations to environmental NGOs or causes; eco-friendly choices in daily life like reusables or transit; documentaries and influencers followed about climate or ecology)",
        "ecological and environmental interests (climate documentaries watched, sustainability influencers followed, environmental causes supported)",
        "the user's environmental engagement (climate-related anxieties or concerns; environmental causes donated to or volunteered with; sustainability conferences or events attended; daily eco-conscious decisions; environmental influencers and media followed)",
    ],
    "Books": [
        "books (genres like mystery or biography, organizing books, book swaps, reading habits)",
        "the user's book habits (genres read like mystery or biography, bookstores frequented, book clubs, home library curation)",
        "books and reading habits (favorite genres like mystery, biography, fantasy, or non-fiction; bookstore preferences like boutique vs chain or thrift vs new; book swaps and book clubs participated in; book collection size and curation; reading frequency and routines; picture books, novels, or e-books)",
        "engagement with books (kinds of books loved, organizing or displaying books, picture books vs novels, reading frequency)",
        "the user's book life (genres regularly read like mystery, biography, fantasy, or non-fiction; bookstores frequented for purchase or browsing whether boutique, chain, or thrift; participation in book swaps, lending circles, or community sharing; home library size and curation choices; ratio of new vs second-hand or print vs digital; reading rituals and frequency)",
    ],
    "Pets": [
        "pets (pet ownership, animals, pet care)",
        "the user's relationship with pets (pet ownership, time spent with cats or dogs, animal shelter volunteering, pet care)",
        "pets and animal companions (pets currently owned like cats, dogs, fish, or reptiles; time spent with pets at home or in the neighborhood; volunteering at animal shelters or rescues; pet care routines and tools; specific affinities for certain breeds or species; childhood or past pets remembered fondly)",
        "pets in the user's life (cats, dogs, or other pets owned or cared for, time with neighborhood pets, animal rescue volunteering)",
        "the user's pet relationships (current and past pets and their breeds or species like cats, dogs, fish, reptiles, or birds; daily pet care routines and feeding; relationships with neighborhood animals or strays; animal shelter or rescue volunteering; species or breed affinities; emotional attachment to specific pets and their roles in family life)",
    ],
    "Science": [
        "science (scientific topics, research, experiments)",
        "the user's interest in science (science museum visits, astronomy newsletters, science podcasts, research topics followed)",
        "science and scientific topics (specific science fields followed like astronomy, biology, physics, or chemistry; popular-science reading from books, magazines, or online; science museum visits and exhibits; podcasts, newsletters, and YouTube channels about science; lay interest in research and discoveries)",
        "scientific topics the user follows (specific fields like astronomy or biology, museums and exhibits, science journalism)",
        "the user's interest in science (specific scientific fields followed in news or media like astronomy, biology, physics, chemistry, or neuroscience; science museums and exhibits visited; podcasts, newsletters, and YouTube channels about science; popular-science books read by accessible authors; lay engagement with research or experiments; science journalism consumed)",
    ],
    "Finance": [
        "finance (money management, investments, savings, spending)",
        "the user's finances (money management, investments, anxieties about financial security, budgeting habits)",
        "finance and money management (savings, investments, and budgeting habits; financial anxieties about income stability or unexpected expenses; risk tolerance for investments like start-ups or markets; spreadsheets and budgeting tools used; multi-currency or international financial concerns; debts, retirement, and long-term financial planning)",
        "money and financial life (income stability concerns, investment choices, spreadsheets for budgeting, attitudes about money)",
        "the user's financial life (money management style and tools like spreadsheets or budgeting apps; investing decisions and risk appetite for stocks, real estate, or start-ups; financial stress sources like income instability, debt, market volatility, or unexpected expenses; saving and spending habits; budgeting routines for short and long term; retirement planning)",
    ],
    "History": [
        "history (historical events, eras, figures)",
        "the user's interest in history (historical landmarks visited, antique maps collected, historical society attendance, archaeology)",
        "history and historical interests (specific eras, regions, or events that fascinate the user; historical landmarks visited; historical society memberships and events; archaeological discoveries and antique objects; antique maps, books, or artifacts collected; reading or media consumption about history)",
        "historical interests (specific historical periods or events that fascinate the user, museums and landmarks visited)",
        "the user's historical interests (eras and regions they study or read about like ancient Rome, medieval Europe, or East Asian dynasties; historical sites visited like castles, ruins, or museums; antique collections or artifacts at home; historical society participation or membership; books, lectures, podcasts, or documentaries about history; archaeological discoveries followed)",
    ],
    "Language": [
        "language (foreign languages, language learning, linguistics)",
        "the user's relationship with language (bilingual or multilingual fluency, languages currently being learned, code-switching habits)",
        "language and languages (native and additional languages spoken; foreign languages currently being learned like Japanese, Spanish, or French; multilingual identity in family or social settings; code-switching and mixing languages in casual speech; bilingual pride and translation work; linguistic curiosities and language learning routines)",
        "languages used and learned (native language, second or third languages, language learning hobbies, accent or dialect pride)",
        "the user's linguistic life (languages fluent in or actively learning like Japanese, Spanish, Mandarin, or French; multilingual everyday communication with family or friends; mixing or code-switching languages in casual speech; language-learning hobbies, resources, apps, or classes; emotional weight of bilingual or multilingual identity; linguistic pride or curiosity)",
    ],
    "Collecting": [
        "collecting (collectibles, memorabilia, hobbies of accumulation)",
        "the user's collections (vintage records, postcards, fridge magnets, novelty mugs, thrift-store finds)",
        "collecting and accumulating things (themed collections kept like vintage records, postcards, magnets, or coins; objects gathered from travels or specific places; thrift store hunts and treasure-finding hobbies; novelty items like mugs or pins; collecting communities or swap groups; how the user displays or stores their collections)",
        "things the user collects (specific objects or themes, where collections are kept, how they're displayed, related communities)",
        "the user's collecting habits (specific objects or themes collected like vintage records, postcards, magnets, coins, or memorabilia; sources of new acquisitions like thrift stores, travel souvenirs, or trade communities; how collections are organized, stored, and displayed; collecting communities or swap groups followed; sentimental value of specific pieces tied to memories)",
    ],
    "Writing": [
        "writing (creative writing, journalism, blogging, essays)",
        "the user's writing practice (journaling, poetry, blogging, zine-making, handwritten letters)",
        "writing as practice or hobby (journaling for personal thoughts and reflection; creative writing like poetry, short stories, or essays; blogs, articles, or zine-making; calligraphy or cursive writing as practice; handwritten letters or birthday cards; writing groups or workshops attended)",
        "the user's writing habits (poetry written occasionally, journals kept regularly, articles or blogs published, creative essays)",
        "the user's writing life (forms of writing practiced like journaling for self-reflection, poetry, blogs, articles, or essays; writing routines and frequency like daily, weekly, or in bursts; creative-writing communities, workshops, or zine-making groups; specific projects worked on like a memoir or novel; calligraphy or handwritten correspondence as ongoing practice)",
    ],
    "Religion": [
        "religion and spirituality (faith, religious practices, beliefs)",
        "the user's religious life (regular church or temple attendance, religious holidays observed, faith-based ceremonies)",
        "religion and spiritual practice (faith tradition the user belongs to or engages with; regular service attendance like church, mosque, temple, or synagogue; religious holidays observed like Christmas, Eid, or Diwali; rituals and ceremonies participated in; faith-based community involvement; reconciliation between faith and other parts of life)",
        "religion and faith (the user's religious tradition, attendance at services, holidays observed, family religious tensions)",
        "the user's religious life (faith tradition, denomination, or non-affiliation across Christianity, Islam, Judaism, Hinduism, Buddhism, or other traditions; service attendance frequency like weekly, monthly, or holiday-only; religious holidays observed and ceremonies attended; sacraments or rituals participated in; faith community ties; spiritual reading and reflection practices)",
    ],
    "Politics": [
        "politics (political views, policy, civic engagement)",
        "the user's political engagement (newspapers read, political causes supported, civic participation, attitudes about discussing politics)",
        "politics and civic engagement (political views and ideological leanings; political news consumption from specific outlets; civic activities like voting, donating, or attending rallies; political causes supported through pins, donations, or advocacy; willingness to discuss politics in personal settings; political stress about national or regional events)",
        "politics and civic life (news regularly read, political pins or causes supported, attitudes toward political discussion)",
        "the user's political engagement (political ideology and party leanings on left-right or other dimensions; news sources read for politics like newspapers, magazines, or podcasts; civic participation like voting, donating, attending rallies, or contacting representatives; political conversations with family or friends and willingness to debate; political worry or stress about national or local events)",
    ],
    "Identity": [
        "personal identity (self-concept, gender, sexuality, cultural identity)",
        "the user's sense of identity (cultural identity navigation, gender or sexual identity, family responses to coming out, identity tensions)",
        "personal identity and self-concept (cultural and ethnic identity navigation across settings; gender identity and expression; sexual orientation and romantic identity; coming-out experiences with family or peers; identity-driven challenges or pride; reconciling cultural traditions with personal identity; identity tensions in mixed environments)",
        "the user's identity (self-concept across cultural, gender, and sexual dimensions; identity-driven challenges or pride)",
        "the user's identity (cultural, ethnic, gender, and sexual identity dimensions and how they intersect; identity-driven life events like coming out, transitioning, or relocating to mixed environments; tensions between expected and authentic identity; cultural identity navigated in workplaces, schools, or family; identity affirmation or pride; identity-driven communities)",
    ],
    "Parenting": [
        "parenting (raising children, parental challenges, family dynamics)",
        "the user's parenting life (raising young children, balancing work with kids, worries about kids' wellbeing, school pressures)",
        "parenting and raising children (the user's role as parent including ages and number of children; parenting challenges like school stress, mental-health worries, or behavior; balancing work and parenting demands; specific concerns about a child's wellbeing or development; family dynamics around parenting; pressure on or guidance for children)",
        "raising children (parental responsibilities, balancing work and parenting, concerns over children's mental health or schooling)",
        "the user's parenting life (children's ages, number, and the user's parenting style; parental stresses and worries about kids' mental or physical health, schooling, or social adjustment; balancing work demands with parenting time; specific child-development concerns like academic performance or peer relationships; family-life dynamics around raising children)",
    ],
    "Gaming": [
        "video gaming (esports, video game genres, online gaming)",
        "the user's video game habits (online multiplayer games, RPG communities, racing sims, esports following, retro gaming)",
        "video gaming and digital game culture (specific game genres played like RPG, shooter, racing, sandbox, or simulation; multiplayer vs single-player preferences; competitive online gaming and esports following; cooperative gaming with friends; mobile, PC, or console platforms; gaming communities and online role-playing groups)",
        "gaming life (regular video games played, online gaming friends, esports interests, kinds of games preferred)",
        "the user's video game habits (genres played like RPG, shooter, simulation, or sandbox; platforms used like PC, console, mobile, or VR; multiplayer vs solo gaming preferences; competitive vs cooperative play styles; specific game series or franchises followed like Final Fantasy or Call of Duty; gaming communities, guilds, or online clans; esports following as spectator sport)",
    ],
    "Design": [
        "design (graphic design, interior design, product design)",
        "the user's design interests (interior design styles, design podcasts, modern furniture, mid-century or minimalist preferences)",
        "design and design appreciation (interior design styles like Japanese minimalist, mid-century modern, Scandinavian, or industrial; graphic design, product design, or design-thinking interests; furniture from specific eras or movements; design publications, podcasts, and shows followed; designers admired; design events or exhibitions attended)",
        "design tastes (interior aesthetics, design podcasts followed, modern vs traditional, furniture and product design)",
        "the user's design tastes (interior aesthetics preferred like minimalist, mid-century modern, Scandinavian, or industrial; specific design movements or eras admired like Bauhaus or Art Deco; furniture and product design choices; design media, podcasts, and publications consumed; designers and design firms followed; design as profession, hobby, or appreciation)",
    ],
    "Dance": [
        "dance (dance styles, performances, dance activities)",
        "the user's dance interests (salsa or ballroom dancing, ballet or hula performances attended, cultural dances)",
        "dance and dance activities (specific dance styles practiced like salsa, ballroom, ballet, hula, or contemporary; cultural or traditional dances participated in; dance classes attended for fun or fitness; dance performances watched live; dance social events; dance as ritual at cultural occasions)",
        "dance habits (styles practiced or learned, attending dance shows, cultural dance traditions)",
        "the user's dance involvement (dance styles practiced or learned like salsa, ballroom, ballet, hula, hip-hop, or contemporary; dance classes attended regularly for fun, fitness, or competition; live dance performances watched at theaters or festivals; cultural dance traditions tied to heritage; dance as social activity, exercise, or art form)",
    ],
    "Discrimination": [
        "experiences with discrimination (bias, exclusion, social injustice)",
        "the user's experiences with discrimination (workplace microaggressions, exclusion in social groups, stereotyping based on background)",
        "experiences with discrimination and bias (workplace microaggressions about background or accent; exclusion in social or professional groups; stereotyping based on identity, gender, or origin; witnessing discrimination of friends or peers; bias in education, housing, or healthcare; emotional impact of recurring or one-off events)",
        "exposure to bias and exclusion (personal experiences of being excluded or stereotyped, witnessing peers face discrimination)",
        "the user's encounters with discrimination (specific incidents of bias or exclusion in workplace, education, or social settings; microaggressions about background, accent, or identity; stereotyping based on identity dimensions; witnessing discrimination of friends or community members; long-term emotional and psychological effects of repeated bias; navigating bias in mixed environments)",
    ],
    "Theater": [
        "theater (plays, performances, acting, stage productions)",
        "the user's theater interests (community theater participation, plays watched, improv comedy, stage productions)",
        "theater and stage performance (community theater participation as actor or audience; specific genres like musicals, drama, comedy, or improv; live plays attended at local or larger venues; classical vs contemporary vs experimental theater; theater clubs and acting groups; theater festivals and conventions)",
        "theater and performing arts (plays attended, community theater involvement, classical or contemporary stage)",
        "the user's theater life (acting in or attending community theater productions; live plays and performances watched at local or larger venues; preferred genres and styles like comedy, drama, musical, or experimental; theater festivals or fringe shows attended; participation in stage productions as actor, crew, or audience; theater clubs joined)",
    ],
    "Theatre": [
        "theater (plays, performances, acting, stage productions)",
        "the user's theatre engagement (kabuki or classical theatre, experimental productions supported, English-language theatre while traveling)",
        "theatre and live stage performance (specific theatre traditions like kabuki, opera, classical, or experimental; small intimate venues vs large productions; English-language vs other-language theatre; theatre attendance while traveling abroad; theatre groups or productions supported; classical drama appreciation)",
        "stagecraft and theatre (live theatrical performances attended, classical theatre traditions, small experimental groups)",
        "the user's theatre engagement (specific theatrical traditions admired like kabuki, opera, classical Shakespeare, or experimental productions; live theatre attendance frequency at large venues or intimate ones; theatre attended while traveling abroad; theatre groups or productions followed or supported financially; traditional vs contemporary stage; specific actors or directors followed)",
    ],
    "Astronomy": [
        "astronomy (stars, planets, telescopes, space)",
        "the user's astronomy interest (stargazing on clear nights, night sky observation, astronomy clubs or events)",
        "astronomy and the night sky (stargazing on clear nights from home or remote locations; specific celestial events observed like meteor showers or eclipses; telescopes and astronomy gear; astronomy newsletters, podcasts, and clubs; observatories visited; planetary or deep-sky observation interests)",
        "interest in astronomy (stargazing routines, night sky photography, astronomy clubs, space observation)",
        "the user's astronomy interest (stargazing routines and dark-sky locations; celestial observation gear like telescopes or binoculars; astronomy clubs or events; reading and media about astronomy; specific objects or events of interest like planets or auroras)",
    ],
    "Shopping": [
        "shopping (purchases, stores, online shopping habits)",
        "the user's shopping habits (window-shopping at malls, thrift vs luxury, large supermarkets vs local markets)",
        "shopping habits and store preferences (favorite store types like boutiques, malls, supermarket chains, or local markets; thrift shopping and second-hand habits; online vs in-person shopping balance; window-shopping and browsing rituals; specific brand vs generic preferences; bulk vs frequent small purchases)",
        "shopping preferences (favorite store types, online shopping use, thrift stores or markets, brand vs generic)",
        "the user's shopping (where they typically shop like malls, boutiques, supermarket chains, local markets, or thrift stores; online vs in-person preferences for convenience or experience; shopping as activity, hobby, or routine chore; brand loyalties and store preferences across categories; window-shopping vs purposeful shopping; bulk vs frequent small purchases)",
    ],
    "Friendship": [
        "friendships (close friends, social connections, friend group activities)",
        "the user's friendships (small or large friend groups, friendships strained by life changes, social-circle adjustments after moves)",
        "friendships and friend-circle dynamics (close friends and inner circle composition; struggle to form friendships in new cities or after moves; friendship strain through life transitions like parenthood or career change; long-distance friendships and how the user maintains them; small selective vs large social groups; recent friendship losses or expansions)",
        "friend connections (close vs casual friends, social adjustments after moves, friendships impacted by parenthood or career)",
        "the user's friend life (close-knit inner circle vs broader social circle; how friendships have evolved over time through life stages and relocations; friend-related emotional events like falling-outs, reconnections, or losses; long-distance friendships maintained through video calls, messaging, or visits; specific friend group activities like book clubs, weekly hangs, or group trips)",
    ],
    "Activism": [
        "activism (advocacy, protests, social causes)",
        "the user's activism (LGBTQ+ rights advocacy, women's rights, public forums, generational tensions in movements)",
        "activism and advocacy (specific causes the user advocates for like LGBTQ+ rights, women's rights, or environmental causes; protests, rallies, or public forums attended; advocacy work alongside daily life or career; generational tensions within movements; movement burnout and pacing of activism; activism as identity)",
        "social and political activism (causes the user advocates for, public forums attended, movement involvement balanced with other duties)",
        "the user's activism (causes advocated for like LGBTQ+ rights, women's rights, environmental, racial, or labor justice; specific advocacy roles or events like organizing rallies, public speaking, or letter-writing; balancing activism with work and family demands; movement involvement over time and burnout cycles; intergenerational tensions or alliances in activism)",
    ],
    "Community": [
        "community (neighborhood, community involvement, local groups)",
        "the user's community involvement (local fiestas, diaspora groups, veteran reunions, neighborhood ties)",
        "community and neighborhood involvement (local fiestas, fundraisers, or community events attended; diaspora groups or ethnic community ties maintained; veterans' or alumni reunions; relationships with neighbors and local groups; community pride or worry about neighborhood changes; civic-life participation)",
        "engagement with community (local fiestas or events, neighborhood pride or worry, ethnic or diaspora community ties)",
        "the user's community ties (local groups joined like neighborhood associations, religious congregations, or cultural clubs; community events attended like fiestas, fundraisers, or fairs; neighborhood relationships and friendships with residents; diaspora or ethnic community involvement abroad; community pride or worry about shifts and changes in the local area)",
    ],
    "Charity": [
        "charity and giving (donations, volunteering, philanthropy)",
        "the user's charitable giving (regular donations to specific causes, charity walks or runs, disaster-relief volunteering)",
        "charity, giving, and philanthropy (regular donations to specific causes like veterans, disaster relief, education, or religious charities; volunteering hands-on at charity events or relief efforts; charity walks, runs, or fundraising events participated in; philanthropic priorities and giving routines; specific organizations supported)",
        "donations and philanthropy (recurring giving to specific organizations, regional charity work, charitable events participated in)",
        "the user's charitable activity (specific organizations donated to regularly like veterans' charities, food banks, or religious organizations; volunteer work at charity events or relief efforts; participation in fundraising like walks, runs, or bake sales; philanthropic giving priorities and recurring vs one-time giving; specific causes supported over years)",
    ],
    "Outdoors": [
        "outdoor activities (hiking, camping, outdoor adventures)",
        "the user's outdoor activity (mountain biking, kayaking, walks in scenic places, weekend bike rides)",
        "outdoor activity and recreation (hiking, biking, kayaking, or other outdoor sports; weekend rides on scenic trails; mountain biking or trail running; walks and runs in countryside or parks; river trips, lake activities, or coastal outings; outdoor adventure planning and gear)",
        "outdoor pursuits (mountain biking, kayaking river trips, countryside cycling, walks in nature)",
        "the user's outdoor recreation (active outdoor pursuits like hiking, biking, kayaking, or trail running; weekend outdoor outings to parks, mountains, or rivers; scenic trails or wilderness exploration on multi-day trips; outdoor adventure as primary hobby or supplement to fitness; outdoor gear ownership; specific trails, regions, or landscapes preferred)",
    ],
    "Beverages": [
        "beverages (drinks, coffee, tea, alcohol)",
        "the user's drink habits (espresso vs chai, sparkling water, fresh lime soda, coffee or tea routines)",
        "beverages and drinks (everyday hot drinks like coffee, espresso, chai, or herbal tea; alcoholic beverages like beer, wine, cocktails, or spirits; non-alcoholic drinks like sparkling water, soda, or juices; specialty or seasonal drinks; beverage routines for morning, afternoon, or evening; favorite cafes or bars)",
        "beverage preferences (morning drinks, evening drinks, festive drinks, casual drinks like soda or water)",
        "the user's drink preferences (hot drinks for morning routines like coffee, espresso, chai, or herbal tea; alcoholic vs non-alcoholic preferences and beverage occasions; specialty drinks like espresso shots, fresh-squeezed juices, or smoothies; specific cafes, bars, or breweries frequented; daily hydration choices like sparkling water, soda, or kombucha; festive drinks for celebrations)",
    ],
    "Business": [
        "business (entrepreneurship, business ventures, professional services)",
        "the user's business life (small online ventures, business anxieties during slow periods, succession planning, eco-tourism start-up)",
        "business and entrepreneurship (small business ventures or side projects run; professional services delivered like consulting or freelancing; business anxieties about cash flow, foot traffic, or seasonality; succession planning and family business dynamics; eco-tourism, retail, or service businesses; pandemic-era business challenges)",
        "business activity (current ventures, anxieties about cash flow, growth strategies, professional services delivered)",
        "the user's business endeavors (current or past business ventures like retail, services, eco-tourism, or freelance consulting; entrepreneurial stress sources like cash flow, foot traffic, or seasonal demand; specific industry the business operates in; succession or growth planning for handing off or scaling; balancing business with family demands or career; pandemic-era pivots)",
    ],
    "Reading": [
        "reading (books, articles, reading habits)",
        "the user's reading habits (mystery novels, second-hand bookstore browsing, book clubs, reading rituals like rainy-day reading)",
        "reading habits and routines (genres of books and articles read regularly; reading rituals like bedtime, rainy days, or commute reading; bookstore visits as activity; book clubs and reading groups participated in; second-hand bookstore browsing; magazines, newspapers, and online articles consumed)",
        "the user's reading (books regularly read, reading rituals, bookstores frequented, group reading activities)",
        "the user's reading habits (when and where the user reads like bedtime, commute, rainy days, or weekends; preferred genres like mystery, literary fiction, or history; preferred formats like print, audiobook, or e-book; book club or reading-group activity; bookstores visited including second-hand or boutique; reading frequency and time spent daily or weekly)",
    ],
    "Communication": [
        "communication (messaging, conversation styles, communication preferences)",
        "the user's communication preferences (WhatsApp for family, video calls with friends, multilingual messaging, regular check-ins with relatives abroad)",
        "communication preferences and habits (specific apps used like WhatsApp, iMessage, or Signal for staying in touch; video calls vs text vs voice messages; frequency of contact with family and friends; multilingual communication with relatives across borders; communication style like brief vs detailed; phone-call vs text preferences)",
        "communication habits (apps used to stay in touch, conversation style, frequency of contact with loved ones across distance)",
        "the user's communication style (preferred messaging apps and platforms like WhatsApp, iMessage, Signal, or Telegram; video calls vs voice messages vs text balance; communication frequency with loved ones daily, weekly, or sporadic; specific multilingual or international communication patterns with family abroad; preference for brevity vs detailed conversations)",
    ],
    "Crafts": [
        "crafts (handmade projects, DIY, craft hobbies)",
        "the user's crafting (woodworking, calligraphy, weekend craft fairs, handmade projects)",
        "craft hobbies and handmade projects (specific crafts practiced like woodworking, calligraphy, knitting, sewing, or paper crafts; weekend craft fairs and markets attended; handmade items created for gifts or personal use; craft materials and tools owned; craft community involvement; specific projects completed or ongoing)",
        "crafting activity (woodworking, knitting or sewing, paper crafts avoided or embraced, craft fairs attended)",
        "the user's craft practice (specific crafts pursued like woodworking, calligraphy, knitting, embroidery, sewing, or paper crafts; craft fairs attended on weekends or holidays; tools, materials, and supplies kept on hand; finished or in-progress craft projects displayed or used; craft community ties through workshops, online groups, or local meetups)",
    ],
    "Architecture": [
        "architecture (buildings, design styles, structures)",
        "the user's architectural interests (modern architecture tours, historical European architecture, architectural history)",
        "architecture and built environment (specific architectural styles admired like modern, gothic, Scandinavian, or contemporary; architectural tours, walks, or exhibits attended; historical European or Asian architecture interests; architecture history reading and media; favorite buildings, cities, or architects)",
        "interest in architecture (specific styles admired, tours and exhibits visited, books or media about architecture)",
        "the user's architectural interests (architectural styles or eras admired like modernist, gothic, art deco, Scandinavian, or contemporary; architectural tours, walks, or open-houses visited; books, podcasts, documentaries, or media about architecture and design; specific buildings, cities, or architects followed; architectural history reading and study)",
    ],
    "Lifestyle": [
        "lifestyle (daily routines, life choices, habits)",
        "the user's lifestyle (everyday rituals like quiet mornings, scented candles, reusable bags, sunny weather outings)",
        "lifestyle habits and daily routines (signature daily rituals like quiet mornings, evening walks, or weekend rest; small everyday objects kept around like reusable bags, scented candles, or backpacks; weather and environment preferences for daily comfort; pace and tempo of daily life; sustainable or simple-living choices)",
        "the user's daily routines (morning habits, things kept around, simple pleasures, sustainable lifestyle choices)",
        "the user's lifestyle (daily and weekly routines that anchor the day; everyday possessions kept close like reusable bags, scented candles, or a worn backpack; small daily comforts and rituals like quiet mornings or evening tea; pace of life preferences from slow-living to high-energy; signature lifestyle markers and aesthetic; sustainable or simple-living habits)",
    ],
    "Transportation": [
        "transportation (vehicles, transit, modes of travel)",
        "the user's transportation choices (public transit vs car, urban cycling, electric vehicles, fuel-efficient cars)",
        "transportation choices and modes (preferred transit modes like cycling, public transit, electric or fuel-efficient cars, or ride-share; commuting style and modes; vehicle ownership and types preferred; environmental considerations in transport; urban vs suburban transit; specific vehicles or routes used regularly)",
        "transportation habits (preferred modes for daily commuting, vehicle ownership, environmental considerations in transport)",
        "the user's transportation life (everyday commute mode like driving, cycling, public transit, or walking; vehicle owned or driven including type and fuel; cycling or public-transit preferences for short or long trips; environmental factors in transit choices like emissions or fuel efficiency; multimodal commuting and seasonal transit changes)",
    ],
    "Cooking": [
        "cooking (recipes, kitchen techniques, meal preparation)",
        "the user's cooking practice (family recipes preserved, plant-based experimentation, soup recipes, kitchenware preferences)",
        "cooking practices and kitchen habits (specific recipes followed or improvised; family recipes preserved across generations; ingredients and cuisines explored at home; cookware and kitchenware preferences like stainless steel or cast iron; experimental cooking with plant-based or international flavors; soups, baking, or specific dishes)",
        "the user as a cook (cooking styles, recipes followed or improvised, cookware preferences, kitchen experimentation)",
        "the user's cooking (recipes used or invented for daily meals or special occasions; signature dishes or cuisines mastered like Italian, Indian, Mexican, or fusion; kitchenware preferences like cast iron, stainless steel, or specialty tools; experimentation with new ingredients or techniques; family or inherited recipes preserved across generations)",
    ],
    "Fitness": [
        "fitness (exercise, workouts, physical training)",
        "the user's fitness routine (morning workouts, jogging, yoga, HIIT vs steady-state, fitness trackers)",
        "fitness routines and physical training (specific workout styles like HIIT, weightlifting, yoga, cardio, jogging, or pilates; gym vs home workouts; group classes or solo training; fitness gear and trackers used like smartwatches or wearables; time-of-day exercise preferences; fitness goals like strength, endurance, or flexibility)",
        "physical training (workout style, fitness goals, time of day for exercise, fitness tech and gear)",
        "the user's fitness life (workout types preferred like HIIT, weightlifting, yoga, cardio, or pilates; training frequency and intensity from daily to occasional; gym vs home or outdoor fitness setting; gear and trackers in use like wearables or smart equipment; fitness goals like strength, endurance, flexibility, or weight management; progress-tracking habits)",
    ],
    "Animals": [
        "animals (wildlife, pets, animal behavior)",
        "the user's interest in animals (zoo and aquarium visits, animal rescue volunteering, marine wildlife rescue)",
        "animals and animal-related interests (zoos and aquariums visited; volunteering at animal rescues or wildlife rehab; specific animals beloved like cats, dogs, or marine wildlife; wildlife observation in nature; animal welfare causes supported; pets and other animal companions)",
        "engagement with animals (pet care, time with neighborhood or shelter animals, wildlife rescue volunteering)",
        "the user's interest in animals (specific animal affinities like cats, dogs, marine life, or birds; zoo or aquarium visits frequently or occasionally; volunteer animal-rescue or rehab work; wildlife observation in nature reserves or while traveling; animal welfare causes followed and supported; pets owned and bonds with neighborhood animals)",
    ],
    "Hobbies": [
        "hobbies (leisure interests, recreational pursuits)",
        "the user's hobbies (woodworking, vintage typewriter collecting, fountain pen collecting, astronomy nights)",
        "hobbies and recreational pursuits (specific hobbies like woodworking, vintage typewriter collecting, fountain pen collecting, or astronomy nights; tools, gear, or collections accumulated for hobbies; hobby community involvement and meetups; hobbies practiced regularly vs occasionally; hobbies that have evolved over time)",
        "personal hobbies (signature recreational pursuits, leisure interests, niche collections, hobby community ties)",
        "the user's hobbies (specific recreational pursuits practiced regularly like woodworking, vintage typewriter collecting, fountain pen collecting, or astronomy nights; gear, tools, and supplies accumulated for hobbies over years; hobby groups, meetups, or events attended; signature hobby identifiers tied to identity; balance between solitary and social hobbies)",
    ],
    "Volunteering": [
        "volunteering (donating time, community service)",
        "the user's volunteering (community events, charity bake sales, wildlife rescue work, grassroots activism)",
        "volunteer work and community service (specific causes the user gives time to like community events, charity bake sales, wildlife rescue, or grassroots activism; volunteer roles and organizations served; hands-on event work; compassion-fatigue from sustained volunteering; volunteering balanced with paid work or family)",
        "volunteer activity (causes the user gives time to, frequency of volunteering, types of work done)",
        "the user's volunteer activity (specific organizations served like community centers, animal shelters, food banks, or schools; specific volunteer roles like event work, mentoring, or hands-on relief; hours and frequency of volunteering weekly or monthly; balancing volunteer commitments with paid work or family; compassion fatigue from sustained service)",
    ],
    "Clothing": [
        "clothing (garments, wardrobe, outfits)",
        "the user's clothing (cotton and linen fabrics, flannel shirts, monochrome styling, festive bangles, regional designers)",
        "clothing and what the user wears (preferred fabrics like cotton, linen, or synthetic; everyday outfit style and color palette; festive or traditional clothing for cultural occasions; specific designers followed regionally or internationally; flannel, monochrome, or patterned wardrobes; casual vs formal balance)",
        "clothing choices (fabrics liked, signature outfits, formal vs casual style, regional clothing preferences)",
        "the user's clothing choices (fabrics liked or avoided like cotton, linen, wool, or synthetics; daily outfit aesthetic from monochrome to colorful or casual to formal; festive or traditional dress for cultural occasions like weddings or holidays; designers followed regionally or internationally; signature clothing markers like jewelry, scarves, or specific accessories worn)",
    ],
    "Weather": [
        "weather (climate, seasons, weather conditions)",
        "the user's weather preferences (sunny days vs overcast, training in rain, weather journals, weather apps)",
        "weather and climate preferences (favorite weather conditions like sunny, overcast, rainy, or snowy; weather-related anxieties from past severe events; weather tracking habits like apps or journals; weather impact on mood, training, or daily activity; specific seasons preferred over others)",
        "the user's relationship with weather (preferred conditions, weather tracking habits, weather-driven mood)",
        "the user's weather attitudes (weather conditions preferred for daily activity or mood like sunny, overcast, rainy, or snowy; weather-tracking habits like apps, journals, or barometers; anxiety or sensitivity to severe weather from past events like hurricanes or wildfires; seasonal preferences across spring, summer, fall, and winter; weather impact on training, walks, or outings)",
    ],
    "Sleep": [
        "sleep (sleep habits, rest, sleep quality)",
        "the user's sleep (regular schedule vs disrupted, insomnia from stress, late-night work calls, sleep difficulties)",
        "sleep habits and patterns (regular vs irregular sleep schedule; insomnia from work stress, deadlines, or late-night calls; sleep disturbances tied to specific life events; bedtime rituals and sleep hygiene; sleep quality concerns; sleep environment like noise or light sensitivity)",
        "sleep habits and quality (when the user sleeps, sleep difficulties tied to stress or schedule, sleep routines)",
        "the user's sleep (typical sleep schedule like early-bird or night-owl; insomnia or disturbance triggers like work stress, late-night calls, or noise; sleep-related anxieties or stress from deadlines or news; bedtime routines like reading, meditation, or screen avoidance; sleep environment preferences like dark, cool, quiet, or with white noise)",
    ],
    "Caregiving": [
        "caregiving (caring for others, nursing, support roles)",
        "the user's caregiving role (caring for aging parents, long-distance caregiving stress, balancing care with work)",
        "caregiving responsibilities (caring for aging parents or relatives in another state or country; long-distance caregiving logistics and stress; caregiver fatigue from sustained role; balancing caregiving with work and family; assisted-living transitions; emotional drain of caregiving)",
        "supporting family members (caring for parents in another state, balancing work and caregiving, emotional drain)",
        "the user's caregiving role (specific relatives cared for like elderly parents, in-laws, or chronically ill family members; long-distance vs co-resident caregiving and the logistics each requires; caregiver burnout from sustained duty; balance between caregiving and other life demands like work and parenting; assisted-living or end-of-life transitions navigated)",
    ],
    "Anxiety": [
        "anxiety (stress, worry, anxious feelings)",
        "the user's anxiety (test anxiety, social anxiety at conferences, on-air mistake anxiety, performance anxiety)",
        "anxiety and anxious feelings (recurring anxieties tied to specific contexts like exams, public speaking, or work performance; physical symptoms like racing heart, sweating, or insomnia; anxiety from public-facing or visible roles; performance and social anxiety; anxiety triggers and somatic patterns)",
        "the user's anxious patterns (specific anxiety triggers, contexts where anxiety surfaces, coping mechanisms)",
        "the user's anxiety patterns (anxiety-triggering situations like exams, public speaking, social events, or work performance; somatic symptoms experienced like racing heart, sweating, insomnia, or stomach discomfort; anxiety from specific contexts like leadership, public-facing roles, or scrutiny; coping strategies tried like therapy, meditation, exercise, or medication)",
    ],
    "MentalHealth": [
        "mental health (psychological wellbeing, mental health conditions)",
        "the user's mental health (imposter syndrome, panic attacks, dissociation episodes, lingering self-doubt from past experiences)",
        "mental health and psychological wellbeing (specific conditions or experiences like imposter syndrome, panic attacks, dissociation, depression, or PTSD; processing past trauma or life events; ongoing mental-health work like therapy or coping strategies; lingering effects of past experiences; mental-health stressors)",
        "mental health experiences (depression episodes, panic attacks, anxiety, self-doubt or imposter feelings)",
        "the user's mental health (specific psychological experiences or conditions like depression, anxiety, panic disorder, PTSD, or imposter syndrome; therapy or coping strategies used like CBT, medication, or mindfulness; mental-health journey over time including episodes and recovery; specific stressors and triggers; recovery or maintenance routines and support systems)",
    ],
    "Mental Health": [
        "mental health (psychological wellbeing, mental health conditions)",
        "the user's mental health (depression after life transitions, panic attacks, social anxiety after fallouts, periods of low mood)",
        "mental health experiences (depression after life transitions like moves or losses; panic attacks tied to specific events; social anxiety after fallouts; periods of low mood or withdrawal; emotional draining from relational events; mental-health journey over time)",
        "mental health experiences (low periods, anxiety triggers, processing emotional fallouts)",
        "the user's mental wellbeing (depression episodes after specific events like moves, breakups, losses, or career setbacks; anxiety triggers like panic attacks, social anxiety, or relational fallouts; periods of low mood or withdrawal; mental-health adjustments to major life transitions; emotional recovery work and support systems used; therapy or self-care during difficult periods)",
    ],
    "Productivity": [
        "productivity (efficiency, getting things done, time management)",
        "the user's productivity habits (planners, productivity apps, calendar tools, paper-and-digital combos, decision fatigue)",
        "productivity and time management (planners and productivity apps used like calendar apps, paper planners, or hybrid systems; scheduling routines for daily or weekly tasks; decision-fatigue management; balancing competing demands like personal hobbies, work, and charity; specific productivity tools and workflows)",
        "productivity systems (digital and paper planners, task management style, productivity-app loyalty)",
        "the user's productivity systems (planners or apps used like Notion, Todoist, calendar tools, or paper bullet journals; scheduling habits for daily, weekly, and monthly planning; productivity workflows for tasks, projects, and goals; balancing many demands across work, family, and hobbies; combining digital and paper tools; specific time-management methods)",
    ],
    "Relaxation": [
        "relaxation (rest, unwinding, leisure)",
        "the user's relaxation habits (afternoon naps, listening to rain sounds, weekend naps, quiet downtime)",
        "relaxation and unwinding rituals (afternoon or weekend naps; ambient sounds like rain or nature for relaxation; quiet downtime in evenings or mornings; specific relaxation environments like a chair, bath, or garden; passive vs active leisure; rest rituals after demanding days or weeks)",
        "relaxation routines (when and how the user winds down, ambient or sensory relaxation, naps and leisure)",
        "the user's relaxation (rest rituals like afternoon naps, weekend lie-ins, or evening downtime; ambient sounds or environments used for unwinding like rain sounds, soft music, or nature; weekend rest patterns like slow Sundays or recovery days; relaxation as restorative practice after demanding weeks; specific spaces or settings preferred for rest)",
    ],
    "Trauma": [
        "trauma (difficult experiences, recovery, healing)",
        "the user's trauma history (workplace violence, political instability, online harassment, past traumatic events with lingering effects)",
        "trauma and traumatic experiences (specific past traumas like workplace violence, political instability, online harassment, or surviving a natural event; lingering anxiety triggers like sirens or codes; somatic responses to specific cues; trauma processing and recovery work; long-term effects on daily life)",
        "trauma and its aftermath (specific past traumas, lingering anxiety, somatic triggers from the past)",
        "the user's trauma history (specific traumatic events with lasting effects like workplace violence, political instability, online harassment, or natural disasters; anxiety triggered by specific reminders like sounds, places, or situations; somatic or emotional residue from past trauma in daily life; ongoing healing and recovery work like therapy or support groups)",
    ],
    "Sustainability": [
        "sustainability (eco-friendly practices, environmental responsibility)",
        "the user's sustainability habits (second-hand shopping, reusable bags and bottles, hybrid car, green energy interest)",
        "sustainability and eco-friendly practices (reusable items like bags, water bottles, and containers; second-hand shopping for clothing, furniture, or goods; hybrid or electric vehicles; green energy adoption; reducing waste and consumption; environmental responsibility in daily decisions)",
        "sustainable practices (everyday eco choices, reusables, second-hand shopping, environmental responsibility)",
        "the user's sustainable practices (reusables in daily life like bags, water bottles, and food containers; second-hand purchases over new for clothing, furniture, or electronics; green-energy or low-emission choices like hybrid cars or solar panels; waste-reduction habits like composting and recycling; sustainable lifestyle commitments and influence on purchases)",
    ],
    "Wellness": [
        "wellness (holistic health, self-care, balanced living)",
        "the user's wellness routines (mindfulness, sauna and cold plunge, evening walks, onsen visits, water bottle hydration)",
        "wellness and holistic self-care (mindfulness practices like meditation or breathwork; recovery routines like saunas, cold plunges, or onsen visits; hydration and nutrition for wellness; evening walks or movement rituals; balanced living between work and rest; spa or retreat visits)",
        "holistic wellness (mindfulness, recovery routines like sauna, time in onsen or spas, balanced living)",
        "the user's wellness life (mindfulness and meditation practices for mental clarity; recovery routines like saunas, cold plunges, or onsen visits; hydration and clean-eating choices throughout the day; movement and walks as low-intensity exercise; balance between rest and activity across the week; specific wellness rituals on weekends or holidays)",
    ],
    "Values": [
        "personal values (beliefs, principles, ethics)",
        "the user's values (respect for elders, kindness and empathy, honesty in relationships, humility)",
        "personal values and principles (core values like respect for elders, kindness, empathy, honesty, humility, or family loyalty; ethical commitments guiding decisions; values inherited from family or culture; values applied to relationships, work, or community; principles in tension with practical demands)",
        "values the user holds (kindness, respect for tradition, honesty, family loyalty, ethical principles)",
        "the user's value system (specific principles guiding behavior like respect for elders, kindness, honesty, humility, or family loyalty; values inherited from family or culture and those developed personally; ethical commitments and how they apply to daily decisions; values across relationships, work, and community engagement; values in tension with practical demands)",
    ],
    "Career": [
        "career (professional path, career goals, job changes)",
        "the user's career (current professional path, career pivots considered, doubts after rejections, pressure from family on career choice)",
        "career and professional path (current and past professional positions; career goals, aspirations, and pivots; job-search experiences including rejections; doubts or uncertainty about career direction; family pressure on career choice; career-related self-doubt or pride; specific industries or roles)",
        "career life (current job, professional aspirations, doubts and rejections, considering career changes)",
        "the user's career (current career stage like early, mid, or late and aspirations going forward; past career pivots or changes considered or made; job-search stresses including rejections and uncertainty; family or self-imposed career pressure to follow specific paths; career-related identity and self-doubt; specific industries or roles)",
    ],
    "Emotions": [
        "emotions (feelings, emotional experiences)",
        "the user's emotional life (emotional numbness, loneliness, emotional weight from job, emotional fallout from divorce)",
        "emotions and emotional life (specific recurring feelings like loneliness, overwhelm, numbness, or peace; emotional residue from major events like divorce, losses, or relocations; emotional patterns in daily life; processing complex emotions; emotional weight of career or family roles)",
        "the user's feelings (recent or recurring emotional states, emotional weight of life events, processing emotions)",
        "the user's emotional patterns (recurring emotional states like loneliness, overwhelm, peace, or numbness; emotional weight of major life events like divorces, losses, or moves; processing emotions over time through journaling, therapy, or conversation; specific feelings tied to specific contexts like work, family, or seasons; emotional residue carried from the past)",
    ],
    "Safety": [
        "safety (security, risk awareness, protective practices)",
        "the user's safety concerns (anxieties about local crime, hypervigilance after wildfire warnings, worries about workplace safety decisions)",
        "safety concerns and security awareness (anxiety about local crime or community safety; hypervigilance after specific events like wildfires or natural disasters; workplace safety responsibilities; coping with unpredictable safety situations; safety planning and protective practices; specific safety incidents in the past)",
        "concerns about safety (personal or community safety, hypervigilance, work-related safety stress)",
        "the user's safety awareness (concerns about personal or community safety like crime, traffic, or natural events; hypervigilance after specific events like wildfires, accidents, or attacks; safety planning and protective routines like home security or emergency kits; work-related safety stress for first responders, healthcare, or leaders; specific safety incidents in the past)",
    ],
    "Mindfulness": [
        "mindfulness (meditation, present-moment awareness)",
        "the user's mindfulness practice (gratitude journals, meditation, slow Sunday mornings, present-moment rituals)",
        "mindfulness and present-moment practice (daily meditation or breathwork sessions; gratitude journals kept regularly; slow Sunday mornings or other mindful pacing; sensory attention practices; mindfulness apps or guided sessions used; specific mindfulness traditions like Vipassana or Zen)",
        "the user's mindfulness habits (daily meditation, gratitude practice, mindful pacing of daily life)",
        "the user's mindfulness practice (specific meditation or mindfulness routines like daily sits, body scans, or guided sessions; gratitude practices like journaling or evening reflection; slow living and mindful pacing through unhurried mornings or breaks; mindfulness traditions or teachers followed like Vipassana, Zen, or specific apps; sensory attention practices)",
    ],
    "Cycling": [
        "cycling (biking, cycling activities, bike gear)",
        "the user's cycling habits (biking to work, weekend rides on scenic trails, urban cycling, casual rides)",
        "cycling and biking (daily commuting by bike; weekend recreational rides on scenic trails; cycling intensity from casual to competitive; specific bike types owned like road, mountain, or city bikes; cycling community involvement; cycling gear and bike maintenance)",
        "biking and cycling (regular bike rides, cycling for fitness vs commute, scenic or urban routes)",
        "the user's cycling (commuter vs recreational riding patterns; weekend or trail cycling on scenic routes; cycling intensity from casual to competitive; specific bike types owned like road, mountain, hybrid, or city bikes; cycling gear like helmets, lights, or computers; cycling community ties through clubs, group rides, or online forums)",
    ],
    "Automotive": [
        "automotive (cars, vehicles, automotive interests)",
        "the user's vehicle interests (mid-size sedans, classic muscle cars, pickup trucks, bicycle customization)",
        "automotive and vehicle interests (specific vehicles owned or driven like sedans, SUVs, pickup trucks, or motorcycles; classic-car appreciation like muscle cars or vintage models; mechanical hobbies or customization work; car shows or motorcycle rallies attended; vehicle reliability and practicality preferences)",
        "vehicles in the user's life (current car, classic-car appreciation, automotive hobbyist interests)",
        "the user's automotive life (current vehicle owned and driven like sedans, hatchbacks, SUVs, or pickup trucks; classic or vintage vehicle interest like muscle cars or vintage models; automotive hobbyist work like restoration, customization, or maintenance; car shows, rallies, or motorsport events attended; specific brands, models, or eras admired)",
    ],
    "Film": [
        "film (movies, cinema, filmmaking)",
        "the user's film interests (Latin American cinema, French New Wave, animated films, blockbuster action)",
        "film and cinema (specific national cinemas followed like Latin American, French, Korean, or Japanese; classic vs contemporary films; animated vs live-action; auteur vs blockbuster preferences; film festivals attended; specific directors or eras followed; theater vs streaming consumption)",
        "film tastes (specific cinema traditions, animated films, classic vs contemporary cinema, foreign-language films)",
        "the user's film tastes (national cinemas and traditions followed like Latin American, French New Wave, Japanese, or Korean cinema; specific directors or movements admired; animated vs live-action preferences across genres; classic vs contemporary cinema balance; film-festival attendance and arthouse theater visits; specific films revisited or admired)",
    ],
    "Socializing": [
        "socializing (social events, gatherings, interactions)",
        "the user's social life (small backyard gatherings, playful banter with friends, community fundraisers, online communities)",
        "socializing and social activity (small backyard gatherings vs large events; community fundraisers or church socials; playful banter and casual humor with friends; online communities for social connection; social-isolation periods balanced with social ones; specific social rituals)",
        "the user's socializing (close-knit gatherings, large social events, social withdrawal periods, online socialization)",
        "the user's social life (gathering size and style preferences from intimate dinners to large events; social events attended like parties, fundraisers, or church socials; community fundraisers or social events as recurring rituals; online vs in-person socializing balance; social ritual and rhythm across weeks and seasons; specific friend or family gatherings preferred)",
    ],
    "Stress": [
        "stress (pressure, stressful situations, coping)",
        "the user's stress experiences (financial-pressure stress, multilingual high-stakes stress, juggling work and volunteering, family expectation stress)",
        "stress and pressure (specific stress sources like financial pressure, family expectations, multilingual high-stakes communication, or balancing work and volunteering; chronic vs acute stress; stress from urban or environmental factors; coping strategies tried; stress impact on daily life)",
        "the user's stress (specific stress sources, chronic vs acute stress, life-balance stress, financial or family pressure)",
        "the user's stress patterns (recurring stress sources like financial pressure, family expectations, work demands, or environmental factors; pressure points in work, family, or daily life that compound; coping strategies used like exercise, therapy, journaling, or social support; stress duration whether chronic or acute; stress intensity and impact on sleep, mood, and health)",
    ],
    "Spirituality": [
        "spirituality (spiritual practices, beliefs, inner life)",
        "the user's spiritual life (personal altar at home, incense for prayer or relaxation, spiritual practices tied to heritage)",
        "spirituality and inner life (personal altars or spiritual corners at home; incense, candles, or sacred objects used; spiritual practices tied to heritage or culture; meditation sites or sacred places; reconciling cultural traditions with personal spiritual beliefs; private spiritual rituals)",
        "the user's inner spiritual life (personal practices, sacred objects kept, reconciling tradition with personal belief)",
        "the user's spiritual life (specific spiritual practices and rituals like meditation, prayer, or seasonal ceremonies; sacred objects or spaces kept at home like altars, incense, or icons; spiritual traditions followed including heritage-linked or chosen paths; reconciling spirituality with cultural or family beliefs that may differ; private spiritual reflection)",
    ],
    "Decorating": [
        "decorating (interior decor, decoration projects)",
        "the user's decorating (scented candles, photo walls, mismatched colorful furniture, cross-stitch samplers)",
        "decorating and interior aesthetics (decor objects displayed like scented candles, photo walls, or themed pieces; mismatched colorful or themed furniture; cross-stitch, embroidery, or handmade samplers; family photo walls or memory displays; seasonal or themed decoration; room aesthetics and atmosphere)",
        "decor and decorating (objects collected for display, decoration style of rooms, seasonal or themed decor)",
        "the user's decorating choices (signature decor pieces like scented candles, artwork, or themed objects; room-by-room aesthetics with distinct vibes; themed or seasonal decoration for holidays or moods; family or memory decor like photo walls or heirloom displays; ambiance and atmosphere through lighting, scent, and color; mismatched or curated styles)",
    ],
    "Loneliness": [
        "loneliness (social isolation, feeling alone)",
        "the user's loneliness (loneliness during pandemic isolation, loneliness despite outward success, isolation in remote work or villages)",
        "loneliness and social isolation (loneliness during pandemic or remote-work isolation; isolation despite outward success; loneliness after life changes like moves, breakups, or losses; specific seasonal or contextual loneliness; loneliness in crowds or at events; processing isolation)",
        "the user's experience of loneliness (recurring loneliness, isolation tied to circumstances or seasons, loneliness in crowds)",
        "the user's loneliness experiences (specific contexts triggering loneliness like remote work, post-move adjustment, or empty-nest periods; isolation periods tied to life transitions like relocation, retirement, or breakups; loneliness felt despite outward social presence or success; coping with isolation through online communities, hobbies, or therapy; seasonal loneliness)",
    ],
    "Bullying": [
        "bullying (harassment, intimidation experiences)",
        "the user's bullying history (school bullying for being different, online harassment from opponents, lingering effects from past bullying)",
        "bullying and harassment experiences (childhood or school bullying for being different; workplace bullying or hostile treatment; online harassment from ideological or other opponents; lasting effects of bullying on confidence; resurfacing memories of past bullying; specific incidents shaping behavior)",
        "bullying and harassment (past experiences of being bullied, online ideological harassment, social exclusion)",
        "the user's bullying history (specific past or ongoing bullying experiences in schools, workplaces, or online communities; school, workplace, or online harassment from peers, colleagues, or strangers; lasting impact on confidence, social anxiety, or trust in others; resurfacing memories triggered by similar dynamics; coping mechanisms developed)",
    ],
    "Comedy": [
        "comedy (humor, comedians, comic content)",
        "the user's comedy interests (live stand-up shows, stand-up comedy circuits followed, comedy with friends, comic content)",
        "comedy and humor (live stand-up comedy shows attended at clubs or large venues; specific comedians followed or admired across generations; stand-up specials watched on streaming platforms like Netflix; comedy podcasts, panels, and shows consumed regularly; humor styles preferred like observational, deadpan, surreal, absurdist, or political; comedy circuits, clubs, and festivals)",
        "comedic interests (stand-up specials and live shows, comedians admired, comedy clubs or circuits)",
        "the user's comedy interests (live stand-up consumption at clubs or arenas; comedians followed across the spectrum from veterans to new acts; stand-up specials watched on streaming services or DVDs; comedy clubs visited locally or while traveling; humor in casual social settings like banter and joke-sharing; comedy podcasts and panel shows)",
    ],
    "Puzzles": [
        "puzzles (puzzle-solving, riddles, brain teasers)",
        "the user's puzzle habits (jigsaw puzzles on rainy days, crossword puzzles, evening puzzle building)",
        "puzzles and brain teasers (jigsaw puzzles built on rainy days or weekends; crossword puzzles solved daily or on weekends; sudoku, logic puzzles, or other brain teasers; puzzle communities or competitions; specific puzzle difficulty preferences; puzzle-solving rituals)",
        "puzzle-solving (kinds of puzzles enjoyed, when they're done, difficulty preference)",
        "the user's puzzle habits (jigsaw or crossword puzzles solved on rainy days, weekends, or daily; puzzle frequency and rituals like nightly downtime or weekly sessions; difficulty preferences from beginner to expert; puzzle gear or specific brands like New York Times crossword or Ravensburger jigsaws; logic puzzles, sudoku, and brain teasers)",
    ],
    "Swimming": [
        "swimming (swimming activities, water sports)",
        "the user's swimming (lake swimming in summer, calm-water swimming, pools and open water, community pools)",
        "swimming and water activities (lake swimming in summer or vacation seasons; pool swimming for fitness, lap practice, or recreation; open-water swimming in oceans, rivers, or specific bodies of water; specific swim training routines like sprints or endurance; community pools vs private pools or memberships; swimming as exercise, leisure, or competitive sport; swim safety preferences)",
        "swimming activity (lakes, pools, ocean; recreational vs competitive; safety preferences in water)",
        "the user's swimming (where the user swims like lakes, pools, oceans, or rivers; swim frequency from daily to seasonal; swim training routines or recreational use like leisure laps or wave-jumping; safety preferences in water like calm areas, lifeguards, or specific depths; specific swim destinations or routines tied to summer or travel)",
    ],
    "Farming": [
        "farming (agriculture, farm work, rural life)",
        "the user's farming (farm work and homesteading, crop disease anxieties, drought-related stress, farming YouTube channels followed)",
        "farming and rural life (farm operation or homesteading practice; specific crops or livestock raised; weather and yield anxieties from drought, flood, or pests; farming community involvement; agricultural lifestyle and rural-village ties; homesteading YouTube channels or media)",
        "farming and rural life (farm operations, weather and yield anxieties, agricultural community ties)",
        "the user's farm life (specific farm operation or homesteading practice including crops or livestock; agricultural concerns like weather, drought, pests, or crop disease and their impact on yields; rural community ties through co-ops, neighbors, or markets; agricultural lifestyle markers like seasonal rhythms; homesteading media or YouTube channels followed)",
    ],
    "Coffee": [
        "coffee (coffee preferences, cafes, brewing)",
        "the user's coffee habits (boutique coffee shop visits, espresso shots, craft coffee, coffee mug collecting)",
        "coffee and coffee culture (boutique coffee shops visited; espresso, drip, pour-over, or other brewing methods preferred; craft coffee from local roasters; coffee mug or accessory collecting; café visits as social or solo ritual; specific coffee origins or roasts)",
        "the user's coffee life (specialty coffee, café culture, brewing at home, coffee accessories)",
        "the user's coffee life (preferred brewing methods like espresso, pour-over, French press, drip, or cold brew; specialty coffee shops frequented for atmosphere or beans; coffee gear and mugs collected over time; coffee as morning or evening ritual; specific origins, roasts, or roasters preferred; café culture and visits while traveling)",
    ],
    "Conflict": [
        "conflict (disagreements, conflict resolution)",
        "the user's conflicts (disagreements with colleagues or relatives, neighbor disputes, roommate tension, escalations into personal insults)",
        "interpersonal conflict and disagreements (specific past conflicts with colleagues, relatives, neighbors, or roommates; recurring tensions with specific people; conflict-resolution attempts and outcomes; conflicts that escalated into personal insults; lasting impact of unresolved conflicts; conflict-driven stress)",
        "the user's conflicts and tensions (specific disagreements, lingering tensions, conflict-driven stress)",
        "the user's conflict experiences (specific disagreements navigated with colleagues, relatives, neighbors, or roommates; recurring tensions in personal or professional life with the same people; conflict-resolution strategies tried like direct conversation, mediation, or distance; long-term effects of past arguments on relationships and emotional state)",
    ],
    "Marriage": [
        "marriage (spousal relationships, weddings, married life)",
        "the user's marriage (spousal communication, marital tension from work schedules, family pressure to marry, marriage transitions)",
        "marriage and married life (current marital status and spouse relationship; marital communication and intimacy; tensions from work schedules, finances, or family; family pressure to marry or have children; marriage transitions like newlywed or empty-nest; cultural expectations about marriage)",
        "married life or marital expectations (spouse interactions, marriage pressure from family, marital tensions)",
        "the user's marital life (spouse dynamics and communication patterns over years; marriage tensions and supports tied to work, finances, or family; family or cultural expectations about marriage like timing or partner choice; marriage transitions over time like newlywed, parenting years, or empty-nest; marital intimacy and conflict cycles)",
    ],
    "Beauty": [
        "beauty (cosmetics, beauty routines, skincare)",
        "the user's beauty habits (premium hair care, perfume, luxury cosmetics, well-reviewed skincare brands)",
        "beauty and personal-care products (premium hair care from European or specific brands; perfume and luxury cosmetics; well-reviewed skincare products; beauty routines for daytime, evening, or special occasions; brand loyalties; attitudes toward synthetic vs natural ingredients)",
        "the user's beauty routine (skincare products, cosmetics, fragrance, hair care, brand preferences)",
        "the user's beauty routine (skincare, makeup, hair, and fragrance products used in daily or weekly rituals; brand preferences across drugstore, mid-range, and luxury; beauty rituals and timing like morning, evening, or special occasion; natural vs synthetic ingredient stance; specific brands or product types loyal to)",
    ],
    "Space": [
        "space (outer space, astronomy, space exploration)",
        "the user's interest in space (space exploration news, NASA missions, astronomy events, space-related media)",
        "outer space and space exploration (space-mission news from agencies like NASA, SpaceX, or ESA; specific missions followed like Mars, Moon, or deep-space; astronomy and cosmology reading; telescope ownership or use; space-related media like documentaries or podcasts; sci-fi or space-history reading)",
        "space-related interests (space agency news, exoplanet discoveries, astronomy and cosmology reading)",
        "the user's space interest (space-mission news followed from agencies like NASA, SpaceX, ESA, or others; astronomy and cosmology reading from books, magazines, or online; specific space topics like exoplanets, black holes, or planetary science; space media and documentaries like Cosmos or NASA mission coverage; sci-fi as adjacent interest)",
    ],
    "Movies": [
        "movies (films, cinema, movie genres)",
        "the user's movie tastes (European art films, superhero blockbusters, classics, animated family films, documentaries)",
        "movies and film consumption (favorite genres like comedy, drama, action, sci-fi, animated, or documentary; classic vs contemporary films; European art films vs American blockbusters; movie nights at home vs theater going; specific franchises or directors followed; foreign-language vs English films)",
        "movie preferences (genres preferred, theaters vs streaming, classic vs contemporary, animated or live-action)",
        "the user's movie tastes (preferred genres like comedy, drama, sci-fi, action, or animated; classic vs contemporary films and how often each is watched; theatrical vs streaming consumption habits; specific directors, franchises, or studios followed; foreign-language film consumption with subtitles or dubs; movie nights at home as ritual)",
    ],
    "Transport": [
        "transport (transportation, transit, vehicles)",
        "the user's transit choices (commuter trains, e-scooter for commute, electric car, cycling to work)",
        "everyday transport and transit (commuter trains, subways, or city buses for daily commutes; e-scooters, bikes, or e-bikes for short trips and last-mile travel; electric or hybrid cars vs gas-powered vehicles; cycling to work in casual clothes; public-transit reliance and specific routes used; environmental factors in transport choice; multimodal transit habits)",
        "the user's everyday transport (cycling, public transit, car or scooter use, environmental considerations)",
        "the user's transport modes (everyday transit choices like driving, cycling, public transit, or walking; vehicle types used like compact cars, hybrids, e-bikes, or electric scooters; commute routine across weekdays and weekends; environmental considerations in transport like emissions or fuel efficiency; specific routes preferred)",
    ],
    "Exercise": [
        "exercise (workouts, physical activity)",
        "the user's exercise (morning walks, evening park walks, short runs, swimming, urban cycling)",
        "exercise and physical activity (morning walks, evening park walks, or short runs; swimming as exercise; urban cycling for fitness; specific workout routines like jogging, running, or yoga; exercise frequency and time-of-day preferences; indoor vs outdoor exercise)",
        "physical activity (regular walks, runs, bike rides, swims; intensity and time-of-day preferences)",
        "the user's exercise (preferred activities like walking, running, swimming, cycling, or hiking; exercise time-of-day like early morning, lunch break, or evening; frequency and intensity from daily moderate to occasional intense; outdoor vs indoor exercise settings; exercise as routine, fitness, or stress relief; specific routines or programs followed)",
    ],
    "DIY": [
        "do-it-yourself (DIY projects, home improvement, handcraft)",
        "the user's DIY habits (well-equipped toolbox, home improvement projects, power tool use, occasional DIY)",
        "do-it-yourself projects and home work (well-equipped toolbox kept for repairs; home improvement projects undertaken; power tools used for builds or repairs; small DIY builds like shelves or birdhouses; handcraft and woodworking projects; specific DIY skills developed)",
        "DIY and home projects (handyman tasks, occasional builds, tools owned, projects completed at home)",
        "the user's DIY work (home repairs and improvements done independently; power tools and toolbox used for projects; specific projects like furniture builds, bookshelves, birdhouses, or wall repairs; DIY frequency from weekly to occasional; skill level from beginner to experienced; tutorials or YouTube channels followed for guidance)",
    ],
    "Burnout": [
        "burnout (work exhaustion, professional fatigue)",
        "the user's burnout (creative burnout from intensive work, teaching burnout, advocacy fatigue, project-driven exhaustion)",
        "burnout and professional fatigue (creative burnout from intensive projects; teaching burnout from back-to-back classes; advocacy burnout from sustained activism; recovery patterns and time needed; burnout cycles tied to specific seasons or roles; lasting effects of burnout)",
        "professional burnout (specific past burnout episodes, role-driven fatigue, recovery time needed)",
        "the user's burnout history (specific past burnout episodes like creative, teaching, advocacy, or caregiving burnout; role-driven fatigue tied to specific careers or projects; burnout patterns over time including triggering periods or seasons; recovery routines tried like sabbaticals, therapy, exercise, or career changes; lasting effects on motivation)",
    ],
    "Celebrations": [
        "celebrations (parties, events, festivities)",
        "the user's celebrations (seasonal traditions, birthday parties, holiday rituals, unconventional holidays)",
        "celebrations and festive events (seasonal traditions like Canada Day, Christmas, or summer holidays; birthday parties with friends; holiday gift-giving and festive meals; unconventional holiday celebrations; specific traditions kept across years; party styles like backyard or formal)",
        "the user's celebrations and festivities (kinds of parties or events thrown or attended, holiday celebrations, traditions)",
        "the user's celebrations (signature holidays observed like Christmas, Diwali, Eid, Lunar New Year, or Canada Day; birthday and anniversary rituals across family and friends; festive traditions kept across years like specific meals or activities; party preferences and styles from intimate to large; gift-giving customs and seasonal celebrations)",
    ],
    "Heritage": [
        "heritage (cultural background, family history, ancestry)",
        "the user's heritage (ancestral sites, cultural background concerns, family village ties, heritage funding involvement)",
        "heritage and ancestry (cultural background and ancestral roots; ancestral sites visited or wished for; family village or homeland ties; heritage funding or community-history involvement; complex feelings about colonial or contested history; heritage-linked emotions like pride or sadness)",
        "the user's relationship with heritage (cultural background, ancestral places, family history pride or pain)",
        "the user's heritage (ancestral background and origins across one or more cultures; ancestral sites or villages tied to family history that may or may not be visitable; heritage projects supported like preservation, oral history, or community archives; heritage-linked emotional weight including pride, sorrow, or complex feelings about colonial or contested history)",
    ],
    "Law": [
        "law (legal matters, legal profession)",
        "the user's legal experiences (legal disputes navigated, courtroom dramas watched, property-law complications, legal-system stress)",
        "law and legal matters (legal disputes navigated like business or property disputes; courtroom dramas as media interest; legal-system stress and complexity; specific legal complications around identity, property, or work; legal profession or legal-adjacent work; legal advocacy)",
        "law-related life (legal matters faced, legal-system navigation, courtroom drama as media interest)",
        "the user's legal experiences (specific legal disputes or matters like business, property, family, or workplace disputes; legal-system navigation including hiring lawyers or self-representation; legal-profession ties as practitioner, family member, or observer; courtroom drama as media interest in shows or podcasts; legal advocacy or rights work)",
    ],
    "Ethics": [
        "ethics (moral reasoning, ethical decisions)",
        "the user's ethical concerns (moral dilemmas at work, ethical conflicts in business, ethical stress around resource allocation)",
        "ethics and moral reasoning (moral dilemmas at work or in business; ethical conflicts between values and practical demands; resource-allocation ethics; ethical stress from specific decisions; principles applied to professional choices; ethical commitments)",
        "ethical decisions (work-related moral conflicts, value-based decisions, ethical stress)",
        "the user's ethics (moral dilemmas faced in work, business, or personal life; ethical principles applied to decisions like sustainability, equity, or honesty; specific ethical conflicts between competing duties or values; values in tension with practical demands like profit, time, or relationships; reflection on moral reasoning and outcomes)",
    ],
    "Journalism": [
        "journalism (news reporting, journalistic work)",
        "the user's journalism (press freedom anxieties, reporting on displaced families, ethical conflicts in coverage, sensational headlines)",
        "journalism and reporting (specific stories covered like displaced families, political corruption, or pandemic coverage; press-freedom anxieties; ethical conflicts in journalism; emotional weight of difficult stories; sensational vs serious coverage tensions; balancing reporting with personal toll)",
        "journalistic work (assignments covered, ethical stresses in reporting, press freedom concerns, coverage choices)",
        "the user's journalism (kinds of stories covered like political, environmental, conflict, or human-interest reporting; press-freedom concerns and threats to journalists in their region; journalistic ethics and responsibility to subjects; emotional fatigue from covering trauma, displacement, or violence; specific assignments and challenges navigated)",
    ],
    "News": [
        "news (news consumption, current events)",
        "the user's news habits (Wall Street Journal, regional South Asian news, political news daily, print vs digital)",
        "news consumption habits (specific news outlets read like Wall Street Journal, BBC, or regional papers; political vs business vs general news preferences; print vs digital news; news frequency and routines; international vs regional vs local news; specific news topics tracked)",
        "current events and news (preferred news outlets, specific topics followed, daily news rituals)",
        "the user's news habits (sources read or watched like Wall Street Journal, BBC, NPR, or regional papers; daily news rituals like morning paper or evening podcast; specific topics followed like business, politics, or international affairs; format preferences like print vs digital, long-form vs headlines; news consumption frequency)",
    ],
    "Organization": [
        "organization (organizing, organization systems)",
        "the user's organization (detailed planners, calendars, organized living spaces, weekly scheduling, photo albums organized)",
        "organization and tidiness (detailed planners and calendars used digitally or on paper; organized living spaces and decluttering routines; weekly scheduling habits; photo album or document organization; organizational systems for time and space; specific tools and apps used)",
        "the user's organizing habits (planners, organized spaces, scheduling systems, photo or document organization)",
        "the user's organizing habits (planners and calendars used digitally or on paper; tidiness routines like weekly cleaning or daily decluttering; scheduling systems for tasks, errands, and goals; photo or document organization in albums, folders, or digital archives; organizational systems for closets, files, and time; productivity workflows)",
    ],
    "Cars": [
        "cars (automobiles, vehicle preferences)",
        "the user's car interests (car shows attended, motorcycle rallies, mid-sized hatchbacks, sentimental project cars)",
        "cars and automobiles (current vehicle owned and driven like sedans, hatchbacks, or pickup trucks; sentimental project cars from the past; car shows or biker rallies attended; classic or vintage vehicle interest; mechanical hobbies or customization; vehicle history and emotional ties)",
        "cars and motors (vehicle owned, car shows, biker rallies, project-car restoration)",
        "the user's car life (current vehicle owned and daily driving habits; past sentimental cars like first cars or project restorations; automotive shows or biker rallies attended; classic or vintage car appreciation across specific eras or makes; mechanical hobbies like restoration, modification, or routine maintenance; emotional ties to specific vehicles)",
    ],
    "Camping": [
        "camping (outdoor camping, camping gear)",
        "the user's camping habits (family camping trips, simple camping, rugged outdoor camping, summer camping, state-park camping)",
        "camping and outdoor stays (family camping trips; rugged outdoor camping in remote areas; state-park or developed-campground camping; camping gear like tents, sleeping bags, or stoves; specific camping locations and seasons; camping as family or solo activity)",
        "camping activity (camping trips taken, gear owned, terrain preferences like rugged or developed campgrounds)",
        "the user's camping (camping styles like family-friendly campgrounds, rugged backcountry, or RV-based; camping locations across state parks, national forests, or private lands; camping seasons preferred like summer or shoulder seasons; camping gear like tents, sleeping bags, stoves, or trailers; camping frequency from yearly to monthly; group size and dynamics)",
    ],
    "Crafting": [
        "crafting (handcraft, making things)",
        "the user's crafting (handmade holiday cards, handwritten birthday cards, homemade gifts, birdhouse building)",
        "crafting and handmade work (handmade holiday or birthday cards crafted; handwritten letters or correspondence; homemade gifts like cards, knit items, or built objects; small woodworking projects like birdhouses; specific craft skills like calligraphy or paper crafts; craft project frequency)",
        "the user's crafting projects (greeting cards, gifts made by hand, small woodworking projects)",
        "the user's crafting (handmade cards or letters for holidays and birthdays; homemade gifts like knit items, baked goods, or built objects; specific craft skills like calligraphy, paper crafting, or woodworking; small projects like building birdhouses or planters; craft tools and materials kept on hand; craft frequency)",
    ],
    # ============================================================
    # PrefEval — snake_case topics. Eval-only; single-entry lists.
    # These descriptors are content-grounded versions derived from sampling
    # actual PrefEval preference statements + multiple-choice options per topic.
    # ============================================================
    "entertain_shows": ["TV shows, movies, and films (TV series and films by genre like reality, crime, horror, sitcom, documentary, animated, superhero; content avoidances based on subject matter like violence or zombies; film attributes like length, foreign-language, CGI; what someone watches for screen entertainment)"],
    "entertain_music_book": ["music, books, and comics (music by genre like pop, indie, punk, rock, classical, world; music formats like vinyl vs streaming; books and novels by genre like fantasy, historical fiction, literary, mystery; graphic novels and comic books; reading habits)"],
    "entertain_sports": ["sports activities and events (participating in sports like parkour, cycling, running, martial arts; attending or supporting sporting events including those involving animals like rodeos or horse racing; high-speed and motor sports; adventure and water sports; following or avoiding pro leagues — covers both doing and watching)"],
    "entertain_games": ["video games (game genres like shooters, RPGs, sandbox, puzzle, platformer; mobile, PC, console, and VR games; multiplayer vs single-player; gameplay features like crafting, microtransactions, dialogue volume, pixel-art style; gaming events and conventions)"],
    "lifestyle_dietary": ["diet and eating habits (dietary patterns like vegan, vegetarian, gluten-free, dairy-free, raw, fermented; food intolerances and allergies; flavor and ingredient preferences like spicy or seafood; what foods to eat or avoid in everyday life)"],
    "lifestyle_beauty": ["beauty and personal care products (skincare and cosmetics; ingredient avoidances like parabens, sulfates, essential oils, synthetic fragrances; perfumes and scents; cruelty-free, indie, or niche brand preferences; haircare and body care)"],
    "lifestyle_fit": ["workouts and exercise routines (workout styles like HIIT, weightlifting, yoga, cardio, running, jogging; group classes vs solo training; choreographed vs freestyle classes; fitness wearables and tracking tech; protein and nutrition for muscle-building — focused on personal exercise regimen)"],
    "lifestyle_health": ["general health and wellness, a mixed bucket (medically- or principle-driven nutrition like raw vegan, ketogenic, whole-foods, low-sodium; holistic and natural remedies vs prescription medicine; supplements vs food sources; meditation and mindfulness practices; product or material sensitivities like synthetic fabrics or essential oils; health-driven avoidance of activities like heights)"],
    "shop_home": ["home furnishings, decor, and interior design (furniture and materials like wicker, rattan, wood, glass, leather; wall treatments like wallpaper, paint, murals; design aesthetics like minimalist vs maximalist or modern vs rustic; plants, finishes, fabrics; smart home devices in living spaces)"],
    "shop_fashion": ["clothing and personal style (clothing materials like wool, leather, synthetics, cotton; sustainable or ethical brands vs fast fashion; footwear like heels vs flats; prints, patterns, and colors; style aesthetics like vintage, retro, trendy; what to wear for occasions)"],
    "shop_motors": ["cars and vehicles to own (manual vs automatic transmission; gas vs electric vs hybrid powertrains; vehicle size and class like compact, sedan, SUV, pickup; interior materials like leather vs cloth; autonomous and assisted-driving features; environmental and emissions concerns when picking a car)"],
    "shop_technology": ["consumer technology and devices (specific brands like Samsung, Apple, Google, Microsoft; device categories like phones, laptops, tablets, smart TVs, wearables; OS and ecosystem preferences like iOS, Android, Linux, ChromeOS; cloud storage stance; foldables; product generations and customization)"],
    "travel_restaurant": ["restaurants and dining out, especially while traveling (which restaurants and cuisines to recommend at a destination; dietary restrictions applied to restaurant choices; street food vs sit-down; vegan, gluten-free, dairy-free, or spice-level considerations framed as where to dine)"],
    "travel_hotel": ["hotels and travel lodging (chain hotels vs boutique or independent stays; location attributes like beachfront vs mountain, downtown vs residential, busy vs quiet streets; eco-friendly or sustainable accommodations; hotel design and atmosphere like minimalist vs ornate)"],
    "travel_activities": ["things to do at a travel destination (sightseeing, tours, outdoor adventures, cultural events while on vacation; preferences and avoidances around heights, water sports, deserts, climbing, adventure sports, or strenuous physical activity specifically when planning a trip)"],
    "travel_transportation": ["transportation while traveling (airlines like budget vs full-service; rental cars including transmission and fuel type; ride-share vs taxi vs public transit at a destination; private drivers and chauffeurs; trains, buses, RVs and campervans for travel)"],
    "education_resources": ["educational resources and study materials (free or open-source vs paid courses; video-based vs text-based content; interactive components and Q&A; account-required platforms; books, tutorials, podcasts — focused on the materials and tools used to learn)"],
    "education_learning_styles": ["how someone prefers to learn (in-person classroom vs online or virtual; structured/guided curriculum vs self-directed pacing; group projects and team-based study vs individual learning; gamified vs traditional methods; storytelling/narrative vs rote memorization; team-building or retreat formats)"],
    "professional_work_location_style": ["work and where to live (job functions or industries to do or avoid like healthcare or social-media management; office layouts like open-plan vs private office; cities and locations to live or relocate based on climate, weather, noise, safety, density, urban vs suburban or rural — covers both work setting and life-location)"],
    "pet_ownership": ["pets and animal companions (preferred species and breeds like dogs, cats, fish, reptiles, birds; pet allergies and tolerances; rescue and adoption vs breeders or pet stores; pet care products like flea treatments, toys, food; living with animals)"],
}


def _select(topic: str, rng: random.Random | None) -> str | None:
    """Pick a descriptor for `topic`. Sample uniformly when `rng` is provided
    (training-time augmentation); use index 0 when `rng` is None
    (deterministic eval).

    Returns None if the topic is unknown or has an empty list.
    """
    descs = TOPIC_DESCRIPTORS.get(topic)
    if not descs:
        return None
    if rng is None:
        return descs[0]
    return rng.choice(descs)


def describe_topic(topic: str, rng: random.Random | None = None) -> str:
    """Return 'topic (descriptor)' if known, else just the topic name.

    Pass `rng` to sample a variant uniformly (use during training); omit to get
    the canonical (index-0) descriptor (use during eval).
    """
    desc = _select(topic, rng)
    if desc and "(" in desc:
        return f"{topic} ({desc.split('(', 1)[1].rstrip(')').strip()})"
    return topic


def topic_with_hint(topic: str, rng: random.Random | None = None) -> str:
    """Render a topic for use inside an instruction template:
    - Known topic: full descriptor with parenthetical.
    - Unknown topic: bare topic name.

    Pass `rng` to sample a variant uniformly (use during training); omit to get
    the canonical (index-0) descriptor (use during eval).
    """
    desc = _select(topic, rng)
    return desc if desc is not None else topic
