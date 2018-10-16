/* tslint:disable */
export function get_qap(arg0: string): QAPJS;

export function get_crs(arg0: QAPJS): CRSJS;

export function prove(arg0: string, arg1: Uint32Array, arg2: QAPJS, arg3: CRSJS): ProofJS;

export function verify(arg0: QAPJS, arg1: CRSJS, arg2: Uint32Array, arg3: ProofJS): boolean;

export function greet(arg0: string, arg1: QAPJS, arg2: CRSJS, arg3: ProofJS, arg4: ProofJS): boolean;

export class ProofJS {
free(): void;

}
export class CRSJS {
free(): void;

}
export class QAPJS {
free(): void;

}
