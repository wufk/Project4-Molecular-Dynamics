#pragma once

namespace MD {
	void MD_init(int ratio, int cellsize);

	void MD_free();

	void MD_run();

	void MD_Loop(int k);
}

