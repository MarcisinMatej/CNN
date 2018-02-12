TODO natrenovat 64x64 spravit graf konvergence pro kazdy attribut(x epocha,y chyba)
	spocitat trenovaci,testovaci,validacni
	na testovacich datach spravit - confusion mattrix (chyby true possitive/false positive)

zmenit label training/testing/validation na samostatny list (nemusia byt po kope)

zmerat rychlost ako je halkiva na velikost batchu : je to jedno, keras si bere 32
zmerat ako rychlo sa generuju virtualne obrazky

prihodit tutorial na spustenie + kniznice a ich verzie

TODO 1 doriesit adresare a cesty k datam
TODO 3 validation data generator
TODO 4 generator expand 
TODO 5 pozriet sa ci sa dajako neda spojit moj generator a Keras (data augmentation)
	possibility 1 : easy, augmet one picture at time
	possibility 2 : najst sposob ako spravit tie labels...moznost aby to bralo ako label id obrazku a podla toho nacitat labels alebo najst sposob ako overridnut label creation 
TODO 6 Spocitat presnost na validation datach
TODO 7 preskumat optimizre ( rmsprop/Adam/Adagrad a ich parametr, mozno spravit dajaku optimalizaciu - simulovane zihanie parametrov)