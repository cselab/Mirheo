#include <logistic.h>

int main() {
	KISS kiss( rand(), rand(), rand(), rand() );
	
	for(int i=0;i<32;i++) printf("%f\n", kiss.get_real());

	return 0;
}
