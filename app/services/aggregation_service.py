# Imports
import torch
import numpy as np
from typing import List, Dict
from sklearn.cluster import KMeans
import torch.nn.functional as F

from config import create_logger

logger = create_logger(__name__)

# Create static class with methods
class AggregationService:

    # FedAvg
    @staticmethod
    def fed_avg(updates: List[Dict], *args, **kwargs) -> Dict[str, torch.Tensor]:
        aggregated: Dict[str, torch.Tensor] = {}
        update_count: int = len(updates)

        for update in updates:
            for k, delta in update.get('weights').items():
                if k not in aggregated:
                    aggregated[k] = torch.zeros_like(delta)
                aggregated[k] += delta / update_count

        return aggregated
    
    # Weighted avg
    @staticmethod
    def weighted_fed_avg(updates: List[Dict], weights: Dict[str, float], *args, **kwargs) -> Dict[str, torch.Tensor]:
        aggregated: Dict[str, torch.Tensor] = {}
        sum_weights: float = sum(weights.values())

        for update in updates:
            client_id = update.get('client_id')
            weight = weights.get(client_id)

            for k, delta in update.get('weights').items():
                if k not in aggregated:
                    aggregated[k] = torch.zeros_like(delta)
                aggregated[k] += weight * delta / sum_weights

        return aggregated
    
    # Multi krum
    @staticmethod
    def m_krum(updates: List[Dict], n_byzantine: int, m: int = None, *args, **kwargs) -> Dict[str, torch.Tensor]:
        
        update_count: int = len(updates)
        if update_count <= 2 * n_byzantine + 2:
            print(f'M-Krum needs at least {2 * n_byzantine + 3} clients for f = {n_byzantine}')
            return AggregationService.fed_avg(updates = updates)
        
        if m is None:
            m = update_count - n_byzantine

        m = min(m, update_count - n_byzantine)
        m = max(m, 1)

        def flatten_weights(weights: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.cat([w.flatten() for w in weights.values()])
        
        model_weights: List[Dict[str, torch.Tensor]] = [_.get('weights') for _ in updates]
        flattened_weights = [flatten_weights(w) for w in model_weights]
        
        # Compute each model's score
        n_closest = update_count - n_byzantine - 2
        scores = []

        for i in range(update_count):
            # Compute euclidian distance with every other model
            distances = []
            for j in range(update_count):
                if i != j:
                    dist = torch.norm(flattened_weights[i] - flattened_weights[j], p = 2)
                    distances.append((dist, j))
            
            # Sort by distance and select n_closest closest
            distances.sort(key = lambda x: x[0])
            closest_distances = [d[0] for d in distances[:n_closest]]
            
            # Score = sum of squared distances towards nearest neighbours
            score = sum([d ** 2 for d in closest_distances])
            scores.append((score, i))
        
        # Sort by increasing value and select m best
        scores.sort(key = lambda x: x[0])
        selected_indices = [idx for _, idx in scores[:m]]
        
        logger.info(f"Multi-Krum: {m}/{update_count} selected models")
        
        # Aggregate m best models with mean (aka fed avg)
        selected_weights = [model_weights[i] for i in selected_indices]
        
        aggregated_weights = {}
        for key in selected_weights[0].keys():
            aggregated_weights[key] = torch.stack([
                weights[key] for weights in selected_weights
            ]).mean(dim=0)
        
        return aggregated_weights
    
    # Norm based aggregation
    @staticmethod
    def norm_based_aggregation(updates: List[Dict], n_exclude: int = 1, *args, **kwargs) -> Dict[str, torch.Tensor]:
        
        n_clients = len(updates)
    
        # If not enough clients, use FedAvg
        if n_clients <= 2 * n_exclude:
            print(f"Not enough clients ({n_clients}) to remove {n_exclude} from each side")
            return AggregationService.fed_avg(updates = updates)
        
        # Convert weights into vectors to compute L2 norm
        def flatten_weights(weights: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.cat([w.flatten() for w in weights.values()])
        
        model_weights: List[Dict[str, torch.Tensor]] = [_.get('weights') for _ in updates]
        norms = []
        for i, weights in enumerate(model_weights):
            flat_weights = flatten_weights(weights)
            l2_norm = torch.norm(flat_weights, p=2).item()
            norms.append((l2_norm, i))
        
        # Sort by norm
        norms.sort(key = lambda x: x[0])
        
        # Remove n_exclude lowest and n_exclude highest
        selected_indices = [idx for _, idx in norms[n_exclude:-n_exclude]]
        
        print(f"Norm-Based: {len(selected_indices)}/{n_clients} selected models")
        print(f"  Removed norms (small): {[f'{n:.2f}' for n, _ in norms[:n_exclude]]}")
        print(f"  Removed norms (large): {[f'{n:.2f}' for n, _ in norms[-n_exclude:]]}")
        print(f"  Kept norms: min={norms[n_exclude][0]:.2f}, max={norms[-n_exclude-1][0]:.2f}")
        
        # Aggregate kept models by mean (aka fed avg)
        selected_weights = [model_weights[i] for i in selected_indices]
        
        aggregated_weights = {}
        for key in selected_weights[0].keys():
            aggregated_weights[key] = torch.stack([
                weights[key] for weights in selected_weights
            ]).mean(dim=0)
        
        return aggregated_weights
    
    # DBAA-FedAvg
    @staticmethod
    def cbaa_fed_avg(updates: List[Dict], quantize: bool = True, n_bits: int = 8) -> Dict[str, torch.Tensor]:
        
        n_clients = len(updates)
        model_weights: List[Dict[str, torch.Tensor]] = [_.get('weights') for _ in updates]
    
        # If not enough updates, use FedAvg
        if n_clients < 3:
            return AggregationService.fed_avg(updates = updates)
        
        # Optionnal weight quantification (from 32-bit to 8-bit)
        if quantize:
            quantized_weights = []
            for weights in model_weights:
                quant_w = {}
                for key, tensor in weights.items():
                    # Symetric quantification
                    scale = tensor.abs().max() / (2 ** (n_bits - 1) - 1)
                    if scale > 0:
                        quant_tensor = torch.round(tensor / scale).clamp(
                            -(2 ** (n_bits - 1)), 
                            2 ** (n_bits - 1) - 1
                        ) * scale
                    else:
                        quant_tensor = tensor
                    quant_w[key] = quant_tensor
                quantized_weights.append(quant_w)
            model_weights = quantized_weights
        
        # Compute centroid (norms mean) for each model
        centroids = []
        for weights in model_weights:
            layer_norms = []
            for tensor in weights.values():
                layer_norms.append(tensor.abs().mean().item())
            centroids.append(np.mean(layer_norms))
        
        centroids_array = np.array(centroids).reshape(-1, 1)
        
        # Clustering K-means with k=2 (normal vs malicious)
        kmeans = KMeans(n_clusters = 2, random_state = 42, n_init = 10)
        labels = kmeans.fit_predict(centroids_array)
        
        # Identify largest cluster (the non malicious one)
        cluster_sizes = [np.sum(labels == 0), np.sum(labels == 1)]
        normal_cluster = 0 if cluster_sizes[0] >= cluster_sizes[1] else 1
        
        # Compute intra-cluster distances
        cluster_0_indices = np.where(labels == 0)[0]
        cluster_1_indices = np.where(labels == 1)[0]
        
        def max_intra_distance(indices, centroids_array):
            if len(indices) <= 1:
                return 0
            center = centroids_array[indices].mean()
            return np.max([abs(centroids_array[i] - center) for i in indices])
        
        D1 = max_intra_distance(cluster_0_indices, centroids_array)
        D2 = max_intra_distance(cluster_1_indices, centroids_array)
        D3 = abs(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])[0]
        
        # Clustering criteria: if D3 > max(D1, D2), use clustering
        if D3 > max(D1, D2):
            # Select models from the normal cluster
            selected_indices = np.where(labels == normal_cluster)[0]
            selected_weights = [model_weights[i] for i in selected_indices]
            
            print(f"CBAA-FedAvg: {len(selected_indices)}/{n_clients} selected models (normal cluster)")
        else:
            # Use all models instead
            selected_weights = model_weights
            print(f"CBAA-FedAvg: Using all models")
        
        # Aggregate swith mean (aka fed avg)
        aggregated_weights = {}
        for key in selected_weights[0].keys():
            aggregated_weights[key] = torch.stack([
                weights[key] for weights in selected_weights
            ]).mean(dim=0)
        
        return aggregated_weights
    
    # Trimmed Mean aggregation method
    @staticmethod
    def t_mean(updates: List[Dict], n_exclude: int = 1, *args, **kwargs) -> Dict[str, torch.Tensor]:
        
        aggregated: Dict[str, torch.Tensor] = {}
        models: List[Dict[str, torch.Tensor]] = [_.get('weights') for _ in updates]

        for key in models[0].keys():
            stacked = torch.stack([weights[key] for weights in models])
            sorted_weights, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_weights[n_exclude:-n_exclude]
            aggregated[key] = trimmed.mean(dim=0)

        return aggregated

    # Robust federated Aggregation
    @staticmethod
    def rfa(updates: List[Dict], norm_type: str = 'l2', *args, **kwargs) -> Dict[str, torch.Tensor]:
        
        models: List[Dict[str, torch.Tensor]] = [_.get('weights') for _ in updates]

        # Compute median for each coordinate
        median_weights = {}
        for key in models[0].keys():
            stacked = torch.stack([weights[key] for weights in models])
            median_weights[key] = torch.median(stacked, dim=0).values

        # Compute distance for each model and median
        def flatten_weights(weights: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.cat([w.flatten() for w in weights.values()])
        
        median_flat = flatten_weights(median_weights)
        distances = []
        
        for i, weights in enumerate(models):
            weights_flat = flatten_weights(weights)
            
            if norm_type == 'l2':
                dist = torch.norm(weights_flat - median_flat, p = 2).item()
            elif norm_type == 'l1':
                dist = torch.norm(weights_flat - median_flat, p = 1).item()
            else:
                raise ValueError(f"norm_type doit être 'l1' ou 'l2', reçu: {norm_type}")
            
            distances.append((dist, i))

        # Compute threshold
        dist_values = [d for d, _ in distances]
        median_dist = np.median(dist_values)
        mad = np.median([abs(d - median_dist) for d in dist_values])  # Median Absolute Deviation
        T = median_dist + 2.5 * mad

        # Filter by threshold
        selected_indices = [idx for dist, idx in distances if dist <= T]

        if len(selected_indices) == 0:
            logger.info(f"RFA: No model under threshold T={T:.4f}, using median")
            return median_weights
        
        # Aggregate with mean for the selected models
        selected_weights = [models[i] for i in selected_indices]
    
        aggregated = {}
        for key in selected_weights[0].keys():
            aggregated[key] = torch.stack([
                weights[key] for weights in selected_weights
            ]).mean(dim = 0)
        
        return aggregated

    # FLTrust
    @staticmethod
    def fl_trust(updates: List[Dict], reference_model: Dict[str, torch.Tensor], *args, **kwargs) -> Dict[str, torch.Tensor]:
        
        update_count: int = len(updates)
        models: List[Dict[str, torch.Tensor]] = [_.get('weights') for _ in updates]

        # Convert to flat for similarity computing
        def flatten_weights(weights: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.cat([w.flatten().cpu() for w in weights.values()])
        
        server_flat = flatten_weights(reference_model)

        # Compute trust scores
        trust_scores = []
    
        for i, weights in enumerate(models):
            weights_flat = flatten_weights(weights)
            
            # Cosinus similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                weights_flat.unsqueeze(0), 
                server_flat.unsqueeze(0)
            ).item()
            
            # Using ReLU
            trust_score = max(0.0, cos_sim)
            
            trust_scores.append(trust_score)
        
        # Normalize scores
        total_trust = sum(trust_scores)
        
        if total_trust == 0:
            print("FLTrust: No positive trust score, using FedAvg")
            trust_scores = [1.0 / update_count] * update_count
        else:
            trust_scores = [score / total_trust for score in trust_scores]
        
        print(f"FLTrust: Trust score - min={min(trust_scores):.4f}, max={max(trust_scores):.4f}, mean={np.mean(trust_scores):.4f}")
        
        # Aggregate with trsuted models
        aggregated_weights = {}
        
        for key in models[0].keys():
            weighted_sum = sum(
                trust_scores[i] * models[i][key] 
                for i in range(update_count)
            )
            aggregated_weights[key] = weighted_sum
        
        return aggregated_weights
    
    # CLRA
    @staticmethod
    def clra(updates: List[Dict], reference_model: Dict[str, torch.Tensor], similarity_threshold: float = 0.5, *args, **kwargs) -> Dict[str, torch.Tensor]:

        models: List[Dict[str, torch.Tensor]] = [_.get('weights') for _ in updates]
        n_clients = len(models)
        layer_names = list(models[0].keys())

        # Agregation
        aggregated: Dict[str, torch.Tensor] = {}
        stats: Dict[str, Dict] = {}
 
        for layer_name in layer_names:
            ref_layer = reference_model[layer_name].float()
    
            similarities: List[float] = []
            accepted_indices: List[int] = []
    
            for client_idx, model in enumerate(models):
                sim = AggregationService._cosine_similarity_layer(model[layer_name], ref_layer)
                similarities.append(sim)
    
                if sim >= similarity_threshold:
                    accepted_indices.append(client_idx)
    
            stats[layer_name] = {
                "similarities": similarities,
                "accepted": accepted_indices,
                "rejected": [i for i in range(n_clients) if i not in accepted_indices],
            }
    
            if not accepted_indices:
                # No model passed the filter
                logger.warning(
                    f"[{layer_name}] Mo model accepted"
                    f"(threshold = {similarity_threshold}). "
                )
                aggregated[layer_name] = ref_layer.clone()
                continue
    
            accepted_layers = torch.stack(
                [models[i][layer_name].float() for i in accepted_indices]
            )  # shape: (n_accepted, *layer_shape)
    
            weights = torch.tensor(
                [similarities[i] for i in accepted_indices],
                dtype = torch.float32,
            )
            # Normalization
            weights = weights / weights.sum()
            # Broadcasting : weights shape → (n_accepted, 1, 1, ...)
            shape = [len(accepted_indices)] + [1] * (accepted_layers.dim() - 1)
            weights = weights.view(shape)
            aggregated_layer = (weights * accepted_layers).sum(dim=0)
    
    
            # Back to origin dtype
            original_dtype = models[0][layer_name].dtype
            aggregated[layer_name] = aggregated_layer.to(original_dtype)
    
            n_accepted = len(accepted_indices)
            n_rejected = n_clients - n_accepted
            if n_rejected > 0:
                logger.info(
                    f"[{layer_name}] {n_accepted}/{n_clients} accepted models, "
                    f"{n_rejected} filtered layer(s) "
                    f"(similarities: {[f'{s:.3f}' for s in similarities]})."
                )
    
        return aggregated

    # Cosine similarity layerwise
    @staticmethod
    def _cosine_similarity_layer(layer_a: torch.Tensor, layer_b: torch.Tensor) -> float:
        a_flat = layer_a.flatten().float().cpu()
        b_flat = layer_b.flatten().float().cpu()
    
        # Check for non null norm
        norm_a = torch.norm(a_flat)
        norm_b = torch.norm(b_flat)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
    
        sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
        return sim