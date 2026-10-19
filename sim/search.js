// One-turn look-ahead for a neural player: play each of our candidate joint choices against the opponent's
// likely joint replies in copies of the battle, and keep the choice with the best health margin afterwards.
//
//   const choice = search(battle, 'p1', [{choice: 'move 1 1, move 2', prior: 0.4}, ...]);
//
// Fair play: pass `foeSets` (our belief: revealed details, the rest from usage; see agent.py) and every
// opponent Pokémon in the copy gets those moves, item, ability and stats in place of its real ones. Without
// it the copy holds their real sets, an ORACLE that is only an upper bound. Either way their pending choice
// is cleared, and their replies come from the greedy agent's ranking (top REPLIES per slot) on the copy.
// Still visible to the copy: which of their six they brought, and their RNG-free state (HP, boosts, field).

const path = require('path');
const {Battle, Dex, toID} = require(path.join(__dirname, '..', 'vendor', 'pokemon-showdown', 'dist', 'sim'));
const {GreedyPlayer} = require('./agents/greedy');

const REPLIES = 3;          // opponent options kept per slot (so up to 9 joint replies)
const WORST_WEIGHT = 0.5;   // payoff = (1 - w) * mean over replies + w * worst reply
const PRIOR_WEIGHT = 0.02;  // a small pull towards the policy's own preference, to break near-ties

function healthLeft(side) {
	return side.pokemon.reduce((sum, p) => sum + (p.fainted ? 0 : p.hp / p.maxhp), 0);
}

const NATURES = Dex.natures;

// A copy of the battle from a detached snapshot, with its own log.
function fresh(snapshot) {
	const copy = Battle.fromJSON(snapshot);
	copy.restart(() => {});
	copy.log = []; copy.inputLog = []; copy.sentLogPos = 0;   // the copies share the snapshot's arrays
	return copy;
}

// Put our belief about the opponent's sets into a copy of the battle.
function believe(battle, foe, foeSets) {
	const bySpecies = new Map(foeSets.map(s => [toID(s.species), s]));
	for (const mon of foe.pokemon) {
		const set = bySpecies.get(toID(mon.set.species)) || bySpecies.get(toID(mon.baseSpecies.baseSpecies));
		if (!set) continue;
		const slots = set.moves.map(name => battle.dex.moves.get(name)).filter(m => m.exists).map(m => {
			const old = mon.baseMoveSlots.find(s => s.id === m.id);
			const maxpp = Math.floor(m.pp * 8 / 5);
			return {move: m.name, id: m.id, pp: old ? old.pp : maxpp, maxpp, target: m.target, disabled: false, used: !!(old && old.used)};
		});
		if (slots.length) { mon.baseMoveSlots = slots; mon.moveSlots = slots.map(s => ({...s})); }
		if (set.item && mon.item) mon.item = toID(set.item);       // an item already used up or lost stays gone
		if (set.ability && !mon.species.isMega) mon.baseAbility = mon.ability = toID(set.ability);
		if (set.nature && set.points) {        // the usage spread, on the current form's base stats
			const nature = NATURES.get(set.nature), base = mon.species.baseStats;
			const hpShare = mon.hp / mon.maxhp;
			for (const key of ['atk', 'def', 'spa', 'spd', 'spe']) {
				let value = base[key] + (set.points[key] || 0) + 20;
				if (nature.plus === key) value = Math.floor(value * 110 / 100);
				if (nature.minus === key) value = Math.floor(value * 90 / 100);
				mon.storedStats[key] = mon.baseStoredStats[key] = value;
			}
			mon.maxhp = mon.baseMaxhp = base.hp + (set.points.hp || 0) + 75;
			if (!mon.fainted) mon.hp = Math.max(1, Math.round(hpShare * mon.maxhp));
		}
	}
}

// The opponent's likely joint choices, from the greedy agent's per-slot ranking of the moves the copy gives them.
function replies(battle, foe) {
	const request = foe.activeRequest;
	if (!request || request.wait || request.forceSwitch || request.teamPreview || !request.active) return null;
	const scorer = Object.create(GreedyPlayer.prototype);
	Object.assign(scorer, {getBattle: () => battle, blind: false, request, claimed: [0, 0]});
	const perSlot = request.active.map((active, slot) => {
		const mon = foe.active[slot];
		if (!mon || mon.fainted || request.side.pokemon[slot].commanding) return ['pass'];
		if (active.moves.length === 1) return ['move 1'];      // locked into one move, or Struggle
		const moves = mon.moveSlots.map((move, i) => ({move: {...move, slot: i + 1}})).filter(m => !m.move.disabled);
		if (!moves.length) return ['move 1'];
		return scorer.rankMoves(slot, moves).slice(0, REPLIES).map(o => o.choice);
	});
	const mega = request.active.findIndex((a, slot) => a.canMegaEvo && foe.active[slot] &&
		battle.dex.items.get(foe.active[slot].item).megaStone);
	const joint = [];
	for (const a of perSlot[0]) {
		for (const b of perSlot[1] || ['pass']) {
			const pair = [a, b].map((c, slot) => slot === mega && c.startsWith('move') ? c + ' mega' : c);
			joint.push(pair.slice(0, request.active.length).join(', '));
		}
	}
	return joint;
}

function search(battle, side, candidates, foeSets) {
	const foeId = side === 'p1' ? 'p2' : 'p1';
	if (candidates.length < 2) return candidates[0].choice;
	// toJSON shares arrays (the log among them) with the live battle: detach, or every copy writes into the real game.
	let snapshot = JSON.parse(JSON.stringify(battle.toJSON()));
	if (foeSets) {
		const believed = fresh(snapshot);
		believe(believed, believed[foeId], foeSets);
		snapshot = JSON.parse(JSON.stringify(believed.toJSON()));
	}
	const view = fresh(snapshot);
	const theirs = replies(view, view[foeId]);
	if (!theirs) return candidates[0].choice;
	const seed = [0, 1, 2, 3].map(() => Math.floor(Math.random() * 0x10000));   // fresh rolls, shared by every cell
	const total = candidates.reduce((s, c) => s + c.prior, 0);
	let best = candidates[0], bestScore = -Infinity;
	for (const ours of candidates) {
		const payoffs = [];
		for (const reply of theirs) {
			const copy = fresh(snapshot);
			copy.resetRNG(seed);
			const [mine, other] = [copy[side], copy[foeId]];
			mine.clearChoice(); other.clearChoice();
			if (!mine.choose(ours.choice) || !other.choose(reply)) continue;
			try {
				copy.commitChoices();
			} catch (err) {
				continue;
			}
			payoffs.push((healthLeft(mine) - healthLeft(other)) / mine.pokemon.length);
		}
		if (!payoffs.length) continue;
		const mean = payoffs.reduce((a, b) => a + b, 0) / payoffs.length;
		const score = (1 - WORST_WEIGHT) * mean + WORST_WEIGHT * Math.min(...payoffs) + PRIOR_WEIGHT * Math.log(ours.prior / total);
		if (score > bestScore) { bestScore = score; best = ours; }
	}
	return best.choice;
}

module.exports = {search, believe, fresh, replies};
