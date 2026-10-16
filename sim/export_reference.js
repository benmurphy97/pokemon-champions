// Export reference tables (Pokémon, moves, items) for the current regulation from Showdown itself,
// so the data always agrees with the simulator.
//
//   node sim/export_reference.js            writes data/reference/{pokemon,moves,items,meta}.json

const fs = require('fs');
const path = require('path');

const SHOWDOWN = path.join(__dirname, '..', 'vendor', 'pokemon-showdown');
const {Dex, toID} = require(path.join(SHOWDOWN, 'dist', 'sim'));

const FORMAT_ID = 'gen9championsvgc2026regmc';
const OUT = path.join(__dirname, '..', 'data', 'reference');

const format = Dex.formats.get(FORMAT_ID);
const dex = Dex.mod(format.mod);

const legalMoves = new Map(dex.moves.all().filter(m => !m.isNonstandard).map(m => [m.id, m]));
const legalItems = dex.items.all().filter(i => !i.isNonstandard);

function learnableMoves(species) {
	// Forms without their own learnset (Megas, cosmetic forms) use the base species'.
	for (const id of [species.id, toID(species.baseSpecies)]) {
		const data = dex.species.getLearnsetData(id);
		if (data && data.learnset) {
			return Object.keys(data.learnset).filter(m => legalMoves.has(m)).map(m => legalMoves.get(m).name).sort();
		}
	}
	return [];
}

const pokemon = {};
for (const s of dex.species.all().filter(s => !s.isNonstandard)) {
	pokemon[s.name] = {
		types: s.types,
		baseStats: s.baseStats,
		abilities: Object.values(s.abilities),
		isMega: !!s.isMega,
		battleOnly: s.battleOnly ? [].concat(s.battleOnly) : undefined,  // in-battle form: never put on a team directly
		megaStone: s.requiredItem || undefined,
		baseSpecies: s.baseSpecies,
		moves: learnableMoves(s),
	};
}

const moves = {};
for (const m of legalMoves.values()) {
	moves[m.name] = {
		type: m.type, category: m.category, basePower: m.basePower, accuracy: m.accuracy,
		pp: m.pp, priority: m.priority, target: m.target,
	};
}

const items = {};
for (const i of legalItems) {
	items[i.name] = i.megaStone ? {megaStone: true, megaEvolves: i.megaEvolves} : {};
}

// Type chart: typeChart[attacking][defending] = damage multiplier (0, 0.5, 1 or 2).
const typeChart = {};
const MULTIPLIER = [1, 2, 0.5, 0]; // damageTaken codes: 0 neutral, 1 weak, 2 resists, 3 immune
for (const attacker of dex.types.all()) {
	typeChart[attacker.name] = {};
	for (const defender of dex.types.all()) {
		typeChart[attacker.name][defender.name] = MULTIPLIER[defender.damageTaken[attacker.name] || 0];
	}
}

const natures = {};
for (const n of dex.natures.all()) natures[n.name] = {plus: n.plus || null, minus: n.minus || null};

fs.mkdirSync(OUT, {recursive: true});
const write = (name, data) => fs.writeFileSync(path.join(OUT, name), JSON.stringify(data, null, 1) + '\n');
write('pokemon.json', pokemon);
write('moves.json', moves);
write('items.json', items);
write('typechart.json', typeChart);
write('natures.json', natures);
write('meta.json', {format: FORMAT_ID, mod: format.mod, species: Object.keys(pokemon).length,
	moves: Object.keys(moves).length, items: Object.keys(items).length});
console.log(`pokemon ${Object.keys(pokemon).length}, moves ${Object.keys(moves).length}, items ${Object.keys(items).length}`);
