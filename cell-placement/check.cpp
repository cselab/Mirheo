#include "/home/ytang/crunch-repo/misc/ytang/ermine/src/ermine.h"

using namespace std;
using namespace ermine;

int main()
{
	ParserLammps<ObjectWithBond>  parser;
	Sandbox<ObjectWithBond>       sandbox;
	ObjectWithBond                RBC;

	RBC = parser.read( "rbc2.lammps" );

	ifstream fic("rbcs-ic.txt");
	while( !fic.eof() ) {
		double cx, cy, cz, dummy;
		Matrix3D r;
		fic >> cx >> cy >> cz;
		if (fic.eof()) break;
		fic >> r(0,0) >> r(0,1) >> r(0,2) >> dummy;
		fic >> r(1,0) >> r(1,1) >> r(1,2) >> dummy;
		fic >> r(2,0) >> r(2,1) >> r(2,2) >> dummy;
		fic >> dummy  >> dummy  >> dummy  >> dummy;
		ObjectWithBond rbc = RBC;
		rbc.rotate( r );
		rbc.moveto( cx, cy, cz );
		sandbox.insert( rbc );
	}
	parser.write( "rbc_system.data", sandbox );
}
