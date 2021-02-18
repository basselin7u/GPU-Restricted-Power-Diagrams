#ifndef H_STATUS_H
#define H_STATUS_H

typedef enum {
    vertex_overflow = 0,
    plane_overflow = 1,
    inconsistent_boundary = 2,
    security_radius_not_reached = 3,
    success = 4,
    needs_exact_predicates = 5,
    empty_cell = 6,
    find_another_beginning_vertex = 7,
	triangle_overflow = 8,
    seed_on_border = 9,
    hessian_overflow = 10
} Status;

#define STATUS_NUM (hessian_overflow+1)

#endif
