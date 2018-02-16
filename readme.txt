TODO natrenovat 64x64 spravit graf konvergence pro kazdy attribut(x epocha,y chyba)
	spocitat trenovaci,testovaci,validacni

zmerat ako rychlo sa generuju virtualne obrazky

TODO 1 doriesit adresare a cesty k datam
TODO 4 generator expand 
TODO 5 pozriet sa ci sa dajako neda spojit moj generator a Keras (data augmentation)
	possibility 1 : easy, augmet one picture at time
	possibility 2 : najst sposob ako spravit tie labels...moznost aby to bralo ako label id obrazku a podla toho nacitat labels alebo najst sposob ako overridnut label creation 
TODO 6 Spocitat presnost na validation datach
TODO 7 preskumat optimizre ( rmsprop/Adam/Adagrad a ich parametr, mozno spravit dajaku optimalizaciu - simulovane zihanie parametrov)

Software versions:
	- Python v 3.5.2
	- Keras 2.0.8
	- Keras backend : Tensorflow 1.2.1
	- GraphViz 2.38.0

Folder Structure:
	CNN(main directory)
	 '-->data_proc
	      '-->config_files
	      '-->data
		    '-->test
		    '-->train
		    '-->validation
   	 '-->figures
   	       '-->confusions
	 '-->model
