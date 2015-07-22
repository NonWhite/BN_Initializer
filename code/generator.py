from utils import *
from data import Data
from random import randint
from model import Model

class Generator :
	def __init__( self , modelobj = None ) :
		self.model = modelobj
	
	def synthethicData( self , modelname ) :
		#modelname = os.path.basename( modelname )
		rows_to_generate = int( GENERATED_DATA * TRAINING_DATA_PERCENTAGE )
		self.generateData( GEN_TRAINING_FILE % modelname , rows_to_generate )
		rows_to_generate = int( GENERATED_DATA * TEST_DATA_PERCENTAGE )
		self.generateData( GEN_TEST_FILE % modelname , rows_to_generate )

	def generateData( self , filename , num_rows ) :
		print "Generating data (%s rows) in %s" % ( num_rows , filename )
		with open( filename , 'w' ) as f :
			f.write( ','.join( self.model.topological ) + '\n' )
			for x in xrange( num_rows ) :
				row = self.generateRow()
				line = [ row[ field ] for field in self.model.topological ]
				f.write( ','.join( line ) + '\n' )

	def generateRow( self ) :
		row = {}
		for field in self.model.topological :
			row[ field ] = self.generateValue( field , row )
		return row
	
	def generateValue( self , field , row ) :
		parents = dict( [ ( f , row[ f ] ) for f in self.model.network[ field ][ 'parents' ] ] )
		values = self.model.data.evaluate( [ field ] )
		probs = []
		for val in values :
			cond_prob = self.model.conditional_prob( val , parents )
			probs.append( ( val[ field ] , int( SIZE_TO_GET_RAND_VALUE * cond_prob ) ) )
		rand = []
		for ( val , q ) in probs :
			for x in xrange( q ) : rand.append( val )
		pos = randint( 0 , len( rand ) - 1 )
		return str( shuffle( rand )[ pos ] )

if __name__ == "__main__" :
	model = Model()
	data = Data( TRAINING_FILE , ommit = [ 'fnlgwt' ] )
	data.calculatecounters()
	model.loaddata( data )
	model.loadmodel( DATA_DIR + 'model1.txt' )
	model.trainmodel()
	generator = Generator( model )
	generator.synthethicData( 'prueba' )
